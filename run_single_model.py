import pandas as pd
import os

from preprocessing.preprocessors import preprocess_data
from model.strategies import strategies_dict
from joblib import Parallel, delayed
from config.config import *
from model.models import *


# paralellise the run
def parallel_run(run_number, data, strategy, unique_trade_dates, rebalance_window, validation_window):
    print(f'\nRUN NUMBER: {run_number}\n')
    run_single_strategy(
        df=data,
        strategy=strategy,
        unique_trade_dates=unique_trade_dates,
        rebalance_window=rebalance_window,
        validation_window=validation_window,
        run_number=run_number,
    )


def run_model(strategy_name: str, number_of_runs: int, timesteps: int):
    """Run a single model."""
    # read and preprocess data
    #preprocessed_path = "done_data.csv"
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
    rebalance_window = 63
    validation_window = 63
    
    strategy = strategies_dict.get(strategy_name)
    if strategy is None:
        raise ValueError(
            f'Strategy name {strategy_name} is not recognised. '
            f'Available options: {list(strategies_dict.keys())}'
        )

    # update timesteps
    strategy = strategy._replace(timesteps=timesteps)

    # Parallel execution
    Parallel(n_jobs=4)(
        delayed(parallel_run)(run_number, data, strategy, unique_trade_dates, rebalance_window, validation_window)
        for run_number in range(1, number_of_runs + 1)
    )


if __name__ == "__main__":
    run_model(
       strategy_name='DDPG_MLP_default_relu_TL1000',
       number_of_runs=30,
       timesteps=3_000,
    )