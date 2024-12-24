import numpy as np
import pandas as pd
import os
from stockstats import StockDataFrame as Sdf
from config import config
from typing import Union


def ensure_folder_exists_in_results(*args):
    """
    Ensures that a folder or a folder and its subfolders
    with the specified names exist in the results directory

    Parameters:
    -----------
    *args
        (Sub)folder names to check or create
    """
    # convert the name(s) to string (in case they are integers)
    folder_names = (str(arg) for arg in args)

    # define the full path to the folders
    results_dir = "results"
    folder_path = os.path.join(results_dir, *folder_names)

    # check if the folders exist and create them if they do not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def data_split(df: pd.DataFrame, start: int, end: int):
    """
    Filters data within specified date ranges, where the start date
    is inclusive and the end date is exclusive.
    
    Dates should be provided in the yyyymmdd format as integers.
    This function is designed to generate training and testing datasets.

    Parameters:
    -----------
    df: pd.DataFrame
        A DataFrame containing cleaned stock data.
    start: int
        The start date of the period to filter, inclusive. Expected format is yyyymmdd.
    end: int
        The end date of the period to filter, exclusive. Expected format is yyyymmdd.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing data only within the specified date range,
        ordered by date and ticker (tic).
        The index is redefined such that each unique date shares the same index value.
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

def add_adjusted_fields(df):
    """
    Calculates the adjusted close price, as well as the open, high, low prices, and volume.

    Parameters:
    -----------
    data: pd.DataFrame
        A pandas DataFrame containing stock data.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame with calculated adjusted close price, open-high-low prices, and volume.
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicators(df):
    """
    Calculates 4 technical indicators: MACD, RSI30, CCI30 and DX30
    using the stockstats package

    Parameters:
    -----------
    data: pd.DataFrame
        A pandas DataFrame containing stock data.

    Returns:
    -------
    pd.DataFrame
        A pandas DataFrame with added technical indicators
    """
    data = df.copy()
    # convert input to stock dataframe
    stock = Sdf.retype(data)

    # add new 'close' column mirroring adjusted close price to enable calculations
    stock['close'] = stock['adjcp']

    # unique tickers (already in alphabetical order)
    unique_tickers = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    for unique_ticker in unique_tickers:
        ## macd
        temp_macd = stock[stock.tic == unique_ticker]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)

    data['macd'] = macd
    data['rsi'] = rsi
    data['cci'] = cci
    data['adx'] = dx

    # impute the missing values at the beginning for RSI, CCI, and DX
    data = data.fillna(method='bfill')
    return data


def add_turbulence(df):
    """
    Adds turbulence index from a pre-calculated DataFrame

    Parameters:
    -----------
    data: pd.DataFrame
        A pandas DataFrame containing stock data

    Returns:
    -------
    pd.DataFrame
        Stock data DataFrame enriched with turbulence index
    """
    turbulence_index = calculate_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate', 'tic']).reset_index(drop=True)
    return df


def calculate_turbulence(df):
    """
    Calculates turbulence index
    
    Parameters:
    -----------
    data: pd.DataFrame
        A pandas DataFrame containing stock data

    Returns:
    -------
    pd.DataFrame
        A pandas DataFrame with turbulence index
    """
    df_price_pivot = df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_dates = df.datadate.unique()

    # start after a year
    start = 252
    turbulence_index = [0] * start

    for i in range(start, len(unique_dates)):
        current_price = df_price_pivot.loc[[unique_dates[i]]]
        price_history = df_price_pivot.loc[:unique_dates[i-1]]
        covariance = price_history.cov()

        # calculate how current prices relate to historical averages
        current_temp = (current_price - np.mean(price_history, axis=0))

        # calculate turbulance
        temp = current_temp.values @ np.linalg.inv(covariance) @ current_temp.values.T

        # temp is floored at zero   
        turbulence_temp = max(0, temp[0][0])
        turbulence_index.append(turbulence_temp)

    turbulence_index = pd.DataFrame({
        'datadate': df_price_pivot.index,
        'turbulence': turbulence_index
    })
    return turbulence_index


def preprocess_data():
    """Data preprocessing pipeline"""
    # read data
    df = pd.read_csv(config.TRAINING_DATA_FILE)

    # filter for data after 2009 only - should not remove anything
    df = df[df.datadate >= 20090000]
    
    # calculate adjusted price, open, high, low and volume
    df_with_adjustments = add_adjusted_fields(df)

    # add technical indicators using stockstats
    df_with_adjustments_and_technicals = add_technical_indicators(df_with_adjustments)

    # add turbulence index
    df_final = add_turbulence(df_with_adjustments_and_technicals)
    return df_final