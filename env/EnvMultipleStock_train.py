import numpy as np
import pandas as pd
import gym
import matplotlib
import matplotlib.pyplot as plt
from gym import spaces
from functions import thread_safe_save_plot, thread_safe_save_csv

matplotlib.use('Agg')

# shares normalisation factor
# 100 shares per trade
HMAX_NORMALISE = 100

# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1_000_000

# total number of stocks in our portfolio
STOCK_DIM = 30

# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 1e-3

# reward (difference between starting and ending total assets balances) is scaled
REWARD_SCALING = 1e-4


class StockEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, model_name: str, run_number: int, day: int = 0, save_fig: bool = False):
        self.day = day
        self.df = df
        self.run_number = run_number
        self.save_fig = save_fig

        # action_space normalisation and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1, shape = (STOCK_DIM,))

        # Shape = 181: [Current Balance] + [prices 1-30] + [current holdings per stock 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low = 0, high = np.inf, shape = (181,))
        
        # filter for a single day in input data
        self.data = self.df.loc[self.day, :] 
        
        # starting state is not terminal
        self.terminal = False

        # initalise state
        self.state = (
            [INITIAL_ACCOUNT_BALANCE] +         # initial account balance
            self.data.adjcp.values.tolist() +   # adjusted stock prices
            [0] * STOCK_DIM +                   # stocks held
            self.data.macd.values.tolist() +    # MACD
            self.data.rsi.values.tolist() +     # RSI
            self.data.cci.values.tolist() +     # CCI
            self.data.adx.values.tolist()       # ADX
        )

        # initialise reward (total accumulated increase in total assets) & cost
        self.reward = 0
        self.cost = 0

        # memorise all total balance changes
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.all_asset_memory = pd.DataFrame()
        
        # model name
        self.model_name = model_name

        # trade counter
        self.trades = 0

        # number of training episodes counter
        self.episode = 1

    def _get_stock_info_from_current_state(self, index: int, kind: str):
        if kind == 'price':
            return self.state[index + 1]
        elif kind == 'holdings':
            return self.state[index + STOCK_DIM + 1]
        else:
            raise ValueError(f'kind must be either price or holdings; received {kind}')
        
    def _update_current_holdings(self, index: int, delta: float):
        current_holdings = self.state[index + STOCK_DIM + 1]
        # check if any holdings-related violation
        if delta < 0 and current_holdings < abs(delta):
            raise ValueError(
                f'Trying to sell more than what is available. '
                f'Stock: #{index}, Current holdings: {current_holdings}, '
                f'Amount trying to be sold: {abs(delta)}.'
            )
        
        # update current holdings
        self.state[index + STOCK_DIM + 1] += delta
        
    def _sell_stock(self, index: int, action: float):
        # check if a positive number of shares is held of a given stock
        current_shares = self._get_stock_info_from_current_state(index=index, kind='holdings')

        if current_shares > 0:
            # number of shares to sell considering current shares as a ceiling (max value)
            number_of_shares_to_sell = min(abs(action), current_shares)
            
            # current price
            current_price = self._get_stock_info_from_current_state(index=index, kind='price')

            # update cash balance (it increases after selling shares)
            self.state[0] += current_price * number_of_shares_to_sell * (1 - TRANSACTION_FEE_PERCENT)

            # update shares owned
            self._update_current_holdings(index, -number_of_shares_to_sell)

            # update total cost
            self.cost += current_price * number_of_shares_to_sell * TRANSACTION_FEE_PERCENT

            # update the number of transactions
            self.trades += 1

    def _buy_stock(self, index: int, action: float):
        # if the action would require to buy more than how much cash
        # is available, buy only as far as the cash goes

        # perform buy action based on the sign of the action
        available_cash = self.state[0]
        current_price = self._get_stock_info_from_current_state(index, 'price')
        # max_amount = available_cash // current_price  # original code
        max_amount = available_cash / (current_price * (1 + TRANSACTION_FEE_PERCENT))

        # number of shares to buy considering cash at hand
        number_of_shares_to_buy = min(max_amount, action)

        if number_of_shares_to_buy > 0:
            # update cash balance
            self.state[0] -= current_price * number_of_shares_to_buy * (1 + TRANSACTION_FEE_PERCENT)

            # update shares owned
            self._update_current_holdings(index, number_of_shares_to_buy)

            # update total cost
            self.cost += current_price * number_of_shares_to_buy * TRANSACTION_FEE_PERCENT

            # update the number of transactions
            self.trades += 1
        
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            print(f'Terminating episode #{self.episode}')
            # define output names (used with csv and png, if enabled)
            start_datadate, end_datadate = self.df.datadate.agg([min, max])
            output_path_and_name = (
                f'results/{self.model_name}/{self.run_number}/account_value_train_'
                f'{self.model_name}_{start_datadate}_{end_datadate}'
            )

            # save account value trajectory of the episode as a png, if enabled
            if self.save_fig:
                plot = plt.figure()
                x = [pd.to_datetime(str(x)) for x in self.df.datadate.unique()]
                plt.plot(x, self.asset_memory, 'r')
                plt.xlabel('date')
                plt.ylabel('Account value')
                plt.title(
                    f'Training | min_date={min(self.df.datadate)}, max_date={max(self.df.datadate)}',
                    fontsize=10,
                )
                plt.xticks(rotation=45, fontsize=8, ha='center')
                plt.tight_layout()
                try:
                    thread_safe_save_plot(plot, f'{output_path_and_name}.png')
                except Exception as e:
                    print(f'Error saving the file: {e}')
                plt.close()

            asset_memory_df = pd.DataFrame(self.asset_memory, columns=[f'episode_{self.episode}'])
            training_dates = self.df.datadate.unique()
            asset_memory_df.insert(0, 'datadate', training_dates)

            self.all_asset_memory = pd.concat(
                [
                    self.all_asset_memory,
                    asset_memory_df,
                ],
                axis=1
            )
            thread_safe_save_csv(   
                df=self.all_asset_memory,
                file_path=f'{output_path_and_name}.csv',
                index=False,
            )
            self.episode += 1
            return self.state, self.reward, self.terminal, {}
        else:
            # normalize actions
            actions = actions * HMAX_NORMALISE
            
            # begin total asset = available balance + stock prices * stock current holdings
            available_balance_begin = self.state[0]
            stock_prices_begin = np.array(self.state[1:(STOCK_DIM + 1)])
            current_holdings_begin = np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
            begin_total_asset = available_balance_begin + stock_prices_begin @ current_holdings_begin

            # order the actions and return their indices
            argsort_actions = np.argsort(actions)
            
            # buy and sell indices
            number_of_sell_actions = len(np.where(actions < 0)[0])
            number_of_buy_actions = len(np.where(actions > 0)[0])

            sell_index = argsort_actions[:number_of_sell_actions]
            buy_index = argsort_actions[::-1][:number_of_buy_actions]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            # available balance and current holdings after taking actions (buying/selling)
            available_balance_end = self.state[0]
            current_holdings_end = np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])

            self.state = (
                [available_balance_end] +
                self.data.adjcp.values.tolist() +
                list(current_holdings_end) +
                self.data.macd.values.tolist() +
                self.data.rsi.values.tolist() +
                self.data.cci.values.tolist() +
                self.data.adx.values.tolist()
            )

            # stock prices reflect 'next state'
            stock_prices_end = np.array(self.state[1:(STOCK_DIM + 1)])
            end_total_asset = available_balance_end + stock_prices_end @ current_holdings_end
            self.asset_memory.append(end_total_asset)
            
            # reward = increase/decrease in total assets
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []

        #initiate state
        self.state = (
            [INITIAL_ACCOUNT_BALANCE] +
            self.data.adjcp.values.tolist() +
            [0] * STOCK_DIM + \
            self.data.macd.values.tolist() +
            self.data.rsi.values.tolist() +
            self.data.cci.values.tolist() +
            self.data.adx.values.tolist() 
        )
        return self.state
    
    def render(self, mode='human'):
        return self.state