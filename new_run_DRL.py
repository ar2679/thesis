import pandas as pd
import os

from typing import Union, Dict
from preprocessing.preprocessors import preprocess_data
from model.strategies import strategies_dict
from config.config import *
from model.models import *


def run_model(
        strategy_names_and_timesteps: Dict[str, Union[int, None]],
        number_of_runs: int,
        ensemble_strategy: str = 'default'):
    """Run an ensemble strategy with the inputted strategies"""

    # read and preprocess data
    preprocessed_path = 'processed_stock_source_data.csv'
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data.to_csv(preprocessed_path)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purposes
    # unique_trade_date -> validation dates (data has one record per asset per date)
    unique_trade_dates = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
    # print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    # 63 = 21 * 3, where 21 is the average number of trading days in a month
    rebalance_window = 63
    validation_window = 63

    strategy_list = []
    for i, strategy_name in enumerate(strategy_names_and_timesteps):
        strategy = strategies_dict.get(strategy_name)
        if strategy is None:
            raise ValueError(
                f'Strategy name {strategy_name} is not recognised. '
                f'Available options: {list(strategies_dict.keys())}'
            )

        # update timesteps
        timesteps = strategy_names_and_timesteps[strategy_name]
        if timesteps is not None:
            strategy = strategy._replace(timesteps=timesteps)
        strategy_list.append(strategy)

    for run_number in range(1, number_of_runs + 1):
        print(f'\nRUN NUMBER: {run_number}\n')
        run_ensemble_strategy(
            df=data,
            strategies=strategy_list,
            unique_trade_dates=unique_trade_dates,
            rebalance_window=rebalance_window,
            validation_window=validation_window,
            run_number=run_number,
            ensemble_strategy=ensemble_strategy,
        )

if __name__ == "__main__":
    run_model(
        strategy_names_and_timesteps = {
            'A2C_MLP_default_leaky_relu_TL140': 20_000,
            'DDPG_MLP_wider_leaky_relu_TL140': 3_000,
            'PPO_MLP_deeper_leaky_relu_TL140': 20_000,
        },
        number_of_runs=30,
        ensemble_strategy='default',
    )
