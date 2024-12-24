# THIS IS THE NEW CODE THAT I NEED TO WORK ON
import sys
path = '/Users/'
if path not in sys.path:
    sys.path.append(path)

import logging
logger = logging.getLogger()

import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Type, List
from stable_baselines.common.vec_env import DummyVecEnv
from model.strategies import Strategy
StrategyList = List[Strategy]

# import helper functions
from preprocessing.preprocessors import data_split, ensure_folder_exists_in_results

# customised environments
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade

# suppress the deprication messages
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# scaling softmax for non-greedy ensemble
def scaled_softmax(x, scale=1.0):
    e_x = np.exp((x - np.max(x)) * scale)
    return e_x / e_x.sum()


def DRL_prediction(
        df,
        model,
        model_name,
        run_number,
        last_state,
        iter_num,
        unique_trade_dates,
        rebalance_window,
        turbulence_threshold,
        initial,
        model_specific_weight: float = 1,
        save_fig: bool = True):
    """makes a prediction based on a trained model"""

    # adjustment to ensure trading commences on first day
    k = 0 if initial or len(last_state) == 0 else 1

    # identifying trading days
    trading_data = data_split(df, start=unique_trade_dates[iter_num - rebalance_window - k], end=unique_trade_dates[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(
        df=trading_data,
        model_name=model_name,
        run_number=run_number,
        initial=initial,
        model_specific_weight=model_specific_weight,
        previous_state=last_state,
        turbulence_threshold=turbulence_threshold,
        save_fig=save_fig,
    )])
    
    # if it's the first trading period (initial=True), initialise the state with zero holdings and the max cash balance
    # Otherwise, carry over the final holdings and cash balance from the previous trading period
    obs_trade = env_trade.reset()

    unique_trading_dates = trading_data.index.unique()
    for i, _ in enumerate(unique_trading_dates):
        action, _ = model.predict(obs_trade)
        obs_trade, *_ = env_trade.step(action)
        # check if it is the penultimate date
        if i == len(unique_trading_dates) - 2:
            last_state = env_trade.render()  # rendering does nothing but returns the last state

    return last_state


def DRL_validation(model, test_data, test_env, test_obs):
    # perform the validation process
    unique_validation_dates = test_data.index.unique()
    for _ in enumerate(unique_validation_dates):
        action, _ = model.predict(test_obs)
        test_obs, *_ = test_env.step(action)


def get_validation_sharpe(
        model_name: str,
        run_number: int,
        validation_window: int,
        validation_start_datadate: int,
        validation_end_datadate: int):
    """Calculates the Sharpe ratio based on validation results"""
    input_path_and_name = (
        f'results/{model_name}/{run_number}/account_value_validation_'
        f'{model_name}_{validation_start_datadate}_{validation_end_datadate}.csv'
    )
    df_total_value = pd.read_csv(input_path_and_name, index_col=0)
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (
        (252 / validation_window) ** 0.5 *
        df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    )
    return sharpe


def get_turbulence_threshold(
        df: pd.DataFrame,
        insample_turbulence: float,
        validation_start_datadate: int,
        validation_window: int,
        STOCK_DIM: int,
        verbose: bool = True):
    """
    Calculates Turbulence threshold based on historical and in-sample data
    The historical data range is defined as the period between the first validation
    date and the same date shifted backward by one validation window. 

    If the mean of the historical data exceeds the 90% quantile of the
    in-sample turbulence data, we consider the current market to be volatile.
    In this case, we set the 90% quantile of the in-sample turbulence data
    as the turbulence threshold, ensuring that the current turbulence
    does not exceed this level. 
    Conversely, if the mean of the historical data is below the 90% quantile, 
    we adjust the turbulence threshold upwards, thereby reducing the risk.
    """
    # get the index of the last stock for the validation start date
    end_date_index = df.index[df["datadate"] == validation_start_datadate].to_list()[-1]
    
    # get the index of the first stock for the validation start date adjusted by the validation window
    start_date_index = end_date_index - validation_window * STOCK_DIM + 1

    # filter for the validation range, drop duplicates and calculate mean
    historical_turbulence_mean = (
        df.loc[start_date_index:end_date_index, ['datadate', 'turbulence']]
          .drop_duplicates()
          ['turbulence'].mean()
    )

    # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
    # then we assume that the current market is volatile,
    # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
    # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
    # if the mean of the historical data is less than the 90% quantile of insample turbulence data
    # then we tune up the turbulence_threshold, meaning we lower the risk
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence, .90)
    turbulence_threshold = (
        insample_turbulence_threshold if historical_turbulence_mean > insample_turbulence_threshold
        else max(insample_turbulence.turbulence)
    )
    if verbose:
        print(f"Turbulence threshold: {turbulence_threshold:.3f}")

    return turbulence_threshold


def run_single_strategy(
        df: pd.DataFrame,
        strategy: Type[Strategy],
        unique_trade_dates: np.ndarray,
        rebalance_window: int,
        validation_window: int,
        run_number: int):
    """
    Single agent strategy
    
    Parameters:
    -----------
    df : pd.DataFrame
        full data set (done_data.csv)
    strategy : Strategy (namedtuple)
        contains the model, its name and the number of timesteps
    unique_trade_date : numpy.ndarray
        (unique) validation dates 
    rebalance_window : int
        number of months to retrain the model
    validation_window : int
        number of months to validation the model and select for trading
    run_number : int
    """
    # check if strat and run number specific folders exist; create them if not
    ensure_folder_exists_in_results(strategy.model_name, run_number)

    print(f"{'=' * 18}Start {strategy.model_name}{'=' * 18}")
    last_state = []

    # number of stocks (unique tickers)
    STOCK_DIM = len(df.tic.unique())

    # in-sample trading date-level turbulence data
    insample_turbulence = df.loc[
        (df.datadate >= 20090000) & (df.datadate < 20151000), ['datadate', 'turbulence']
    ].drop_duplicates()

    start = datetime.now()
    for i in range(rebalance_window + validation_window, len(unique_trade_dates), rebalance_window):
        print("=" * 70)
        print(f"Starting time: {start}")
        
        validation_start_date_index = i - rebalance_window - validation_window
        validation_start_datadate = unique_trade_dates[validation_start_date_index]
        
        # initial state is empty (ie no holdings, max cash balance)
        if validation_start_date_index == 0:
            initial = True
        else:
            # previous state
            initial = False

        # fix threshold
        turbulence_threshold = 140

        # the below is for switching over to the quantile-based dynamic threshold
        # establish turbulence threshold to avoid trading in volatile environments
        # turbulence_threshold = get_turbulence_threshold(
        #    df=df,
        #    insample_turbulence=insample_turbulence,
        #    validation_start_datadate=validation_start_datadate,
        #    validation_window=validation_window,
        #    STOCK_DIM=STOCK_DIM,
        #    verbose=True,
        # )

        ############## Environment Setup ##############
        # Training environment (start is included, end is excluded)
        training_data = data_split(df, start=20090000, end=validation_start_datadate)
        env_train = DummyVecEnv([lambda: StockEnvTrain(
            df=training_data, model_name=strategy.model_name, run_number=run_number, save_fig=True
        )])

        ## Validation environment (start is included, end is excluded)
        validation_data = data_split(
            df,
            start=validation_start_datadate,
            end=unique_trade_dates[validation_start_date_index + validation_window]
        )
        env_val = DummyVecEnv([lambda: StockEnvValidation(
            df=validation_data,
            model_name=strategy.model_name,
            run_number=run_number,
            turbulence_threshold=turbulence_threshold,
            save_fig=True,
        )])
        
        obs_val = env_val.reset()

        ############## Training and Validation starts ##############
        print(f"\n{'=' * 10}Model training from 20090000 to {validation_start_datadate} (excl.){'=' * 10}")
        print(f"{'=' * 16}{strategy.model_name} Training{'=' * 16}")
        
        # setting up the model
        trained_model_long_name = (
            f"Training_{strategy.model_name}_TS{strategy.timesteps / 1000:.0f}"
            f"K_until_{validation_start_datadate}_run_number_{run_number}"
        )
        model = strategy.model(
            env_train,
            model_name=trained_model_long_name,
            seed=run_number,
            timesteps=strategy.timesteps
        )  

        print(
            f"\n{'=' * 10}{strategy.model_name} Validation from {validation_start_datadate} "
            f"to {unique_trade_dates[validation_start_date_index + validation_window]} (excl.){'=' * 10}"
        )
        DRL_validation(model=model, test_data=validation_data, test_env=env_val, test_obs=obs_val)

        sharpe = get_validation_sharpe(
            model_name=strategy.model_name,
            run_number=run_number,
            validation_window=validation_window,
            validation_start_datadate=validation_data.datadate.min(),
            validation_end_datadate=validation_data.datadate.max()
        )
        print(f"{strategy.model_name} Sharpe Ratio: {sharpe:.3f}")

        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print(
            f"\n{'=' * 10}Trading from: "
            f"{unique_trade_dates[validation_start_date_index + validation_window]} "
            f"to {unique_trade_dates[i]}{'=' * 10}"
        )
        last_state = DRL_prediction(
            df=df,
            model=model,
            model_name=strategy.model_name,
            run_number=run_number,
            last_state=last_state,
            iter_num=i,
            unique_trade_dates=unique_trade_dates,
            rebalance_window=rebalance_window,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
            save_fig=True,
        )
        print(f"{'=' * 18}Trading Done{'=' * 18}\n")

    end = datetime.now()
    print(f"{strategy.model_name} Strategy took: ", (end - start).seconds / 60, " minutes")


def run_ensemble_strategy(
        df,
        strategies: StrategyList,
        unique_trade_dates: np.ndarray,
        rebalance_window: int,
        validation_window: int,
        run_number: int,
        ensemble_strategy: str = 'default'):
    """Ensemble Strategy that combines multiple models"""
    # check if strategy and run number specific folders exist; create them if not
    for rn in range(run_number):
        ensure_folder_exists_in_results(f'ensemble_{ensemble_strategy}', rn + 1)
    
    sharpe_dict = {}
    trained_model_dict = {}
    for strategy in strategies:    
        sharpe_dict[strategy.model_name] = []
        trained_model_dict[strategy.model_name] = None
    print(f"{'=' * 18}Start Ensemble Strategy{'=' * 18}")
    
    last_state_ensemble = []
    model_use = []

    # number of stocks (unique tickers)
    STOCK_DIM = len(df.tic.unique())

   # in-sample trading date-level turbulence data
    insample_turbulence = df.loc[
        (df.datadate >= 20090000) & (df.datadate < 20151000), ['datadate', 'turbulence']
    ].drop_duplicates()

    start = datetime.now()
    for i in range(rebalance_window + validation_window, len(unique_trade_dates), rebalance_window):
        print("=" * 70)
        print(f"Starting time: {start}")

        validation_start_date_index = i - rebalance_window - validation_window
        validation_start_datadate = unique_trade_dates[validation_start_date_index]
        
        # initial state is empty (ie no holdings, max cash balance)
        if validation_start_date_index == 0:
            initial = True
        else:
            # previous state
            initial = False

        turbulence_threshold = 140

        # the below is for switching over to the quantile-based dynamic threshold
        # establish turbulence threshold to avoid trading in volatile environments
        #turbulence_threshold = get_turbulence_threshold(
        #    df=df,
        #    insample_turbulence=insample_turbulence,
        #    validation_start_datadate=validation_start_datadate,
        #    validation_window=validation_window,
        #    STOCK_DIM=STOCK_DIM,
        #    verbose=True,
        #)

        # create training data (start is included, end is excluded)
        training_data = data_split(df, start=20090000, end=validation_start_datadate)

        # create validation data (start is included, end is excluded)
        validation_data = data_split(
            df,
            start=validation_start_datadate,
            end=unique_trade_dates[validation_start_date_index + validation_window],
        )
        
        for strategy in strategies:
            # set up training environment
            env_train = DummyVecEnv([lambda: StockEnvTrain(
                df=training_data,
                model_name=strategy.model_name,
                run_number=run_number,
                save_fig=True,
            )])

            ## set up validation environment
            env_val = DummyVecEnv([lambda: StockEnvValidation(
                df=validation_data,
                model_name=strategy.model_name,
                run_number=run_number,
                turbulence_threshold=turbulence_threshold,
                save_fig=True,
            )])
            obs_val = env_val.reset()
            
            # set up the model & start training it
            print(f"\n{'=' * 10}Model training from 20090000 to {validation_start_datadate} (excl.){'=' * 10}")
            print(f"{'=' * 16}{strategy.model_name} Training{'=' * 16}")
            
            trained_model_long_name = (
                f"Training_{strategy.model_name}_TS{strategy.timesteps / 1000:.0f}K_until_{validation_start_datadate}"
            )
            model = strategy.model(
                env_train,
                model_name=trained_model_long_name,
                seed=run_number,
                timesteps=strategy.timesteps
            )
            trained_model_dict[strategy.model_name] = model

            # validation starts
            print(
                f"\n{'=' * 10}{strategy.model_name} Validation from {validation_start_datadate} "
                f"to {unique_trade_dates[validation_start_date_index + validation_window]} (excl.){'=' * 10}"
            )
            DRL_validation(model=model, test_data=validation_data, test_env=env_val, test_obs=obs_val)
            sharpe = get_validation_sharpe(
                model_name=strategy.model_name,
                run_number=run_number,
                validation_window=validation_window,
                validation_start_datadate=validation_data.datadate.min(),
                validation_end_datadate=validation_data.datadate.max()
            )
            sharpe_dict[strategy.model_name].append(sharpe)

        # order strategies by their latest sharpe values (descreasing)
        ordered_sharpes = sorted(sharpe_dict.items(), key=lambda x: x[1][-1], reverse=True)

        # apply ensemble logic
        if ensemble_strategy == 'default':  # greedy
            model_ensemble = trained_model_dict[ordered_sharpes[0][0]]  # picking the model with the highest sharpe
            model_use.append(ordered_sharpes[0][0])
            last_state_ensemble = DRL_prediction(
                df=df,
                model=model_ensemble,
                model_name=f"ensemble_{ensemble_strategy}",
                run_number=run_number,
                last_state=last_state_ensemble,
                iter_num=i,
                unique_trade_dates=unique_trade_dates,
                rebalance_window=rebalance_window,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
                save_fig=True,
            )
            print(f"{'=' * 18}Trading Done{'=' * 18}\n")
        elif ensemble_strategy == 'non-greedy':
            initial = False
            # calculate weights for each strategy
            weights = scaled_softmax(pd.DataFrame(sharpe_dict).iloc[-1], scale=3)
            end_states = []
            for strategy in strategies:
                model_specific_weight = weights[strategy.model_name]
                if len(last_state_ensemble) == 0:
                    initial = True
                    strategy_end_state = DRL_prediction(
                        df=df,
                        model=trained_model_dict[strategy.model_name],
                        model_name=f"ensemble_{ensemble_strategy}_{strategy.model_name}",
                        run_number=run_number,
                        last_state=[],
                        iter_num=i,
                        unique_trade_dates=unique_trade_dates,
                        rebalance_window=rebalance_window,
                        turbulence_threshold=turbulence_threshold,
                        initial=initial,
                        model_specific_weight=model_specific_weight,
                        save_fig=True,
                    )
                    end_states.append(strategy_end_state)
                else:
                    # calculating cash balance allocated to strategy
                    starting_state = last_state_ensemble.copy()
                    starting_state[0] = starting_state[0] * model_specific_weight
                    # calculating holdings allocated to strategy
                    holdings = np.array(starting_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
                    starting_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)] = list(holdings * model_specific_weight)
                
                    print(f'Running: {strategy.model_name} | Weight: {model_specific_weight:.3f} | Cash: {starting_state[0]}')
                    strategy_end_state = DRL_prediction(
                        df=df,
                        model=trained_model_dict[strategy.model_name],
                        model_name=f"ensemble_{ensemble_strategy}_{strategy.model_name}",
                        run_number=run_number,
                        last_state=starting_state,
                        iter_num=i,
                        unique_trade_dates=unique_trade_dates,
                        rebalance_window=rebalance_window,
                        turbulence_threshold=turbulence_threshold,
                        initial=initial,
                        save_fig=True,
                    )
                    end_states.append(strategy_end_state)

            # combine the end states of the strategies
            last_state_ensemble = []
            for idx, _ in enumerate(strategy_end_state):
                # aggregate if idx is cash balance or stock holdings
                if idx in [0] + list(range((STOCK_DIM + 1), (STOCK_DIM * 2 + 1))):
                    last_state_ensemble.append(sum(es[idx] for es in end_states))
                else:
                    last_state_ensemble.append(strategy_end_state[idx])
            output_path_and_name = (
                f'results/ensemble_{ensemble_strategy}/{run_number}/last_combined_state_trade_'
                f'ensemble_{ensemble_strategy}_{unique_trade_dates[validation_start_date_index + validation_window]}_'
                f'{unique_trade_dates[i] - 1}.csv'
            )
            pd.DataFrame(last_state_ensemble).to_csv(output_path_and_name)
    end = datetime.now()
    pd.DataFrame(sharpe_dict).to_csv('sharpe_output.csv')
    pd.DataFrame(model_use).to_csv('model_use.csv')
    print(f"Ensemble ({ensemble_strategy}) Strategy took: ", (end - start).seconds / 60, " minutes")





















##### --------------------------------- DONT TOUCH IT

def run_ensemble_strategy_legacy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []

    model_use = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1

        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        #historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]


        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", 20090000, "to ",
              unique_trade_date[i - rebalance_window - validation_window])
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")
        print("======A2C Training========")
        model_a2c = train_A2C(env_train, model_name="A2C_30k_dow_{}".format(i), timesteps=30000)
        print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        print("A2C Sharpe Ratio: ", sharpe_a2c)

        print("======PPO Training========")
        model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=100000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)

        print("======DDPG Training========")
        model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=10000)
        #model_ddpg = train_TD3(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=20000)
        print("======DDPG Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = get_validation_sharpe(i)

        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)

        # Model Selection based on sharpe ratio
        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
            model_ensemble = model_a2c
            model_use.append('A2C')
        else:
            model_ensemble = model_ddpg
            model_use.append('DDPG')
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        #print("Used Model: ", model_ensemble)
        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        # print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
