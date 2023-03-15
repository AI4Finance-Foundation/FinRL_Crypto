"""This script is responsible for processing data using the BinanceProcessor class and saving the results to disk. It
first prints the configuration variables, then processes the data and saves the resulting dataframe, price array,
tech array and time array.

Attributes:
    TICKER_LIST (list): List of tickers to process.
    TECHNICAL_INDICATORS_LIST (list): List of technical indicators to calculate.
    TIMEFRAME (str): Timeframe of the data.
    trade_start_date (str): Start date for trading data.
    trade_end_date (str): End date for trading data.
    no_candles_for_train (int): Number of candles for training.

Functions: main(): main function which runs the script. print_config_variables(): Print the current configuration
variables process_data(): process data using the BinanceProcessor class and return the dataframe, price array,
tech array and time array. save_data_to_disk(data_from_processor, price_array, tech_array, time_array): save the
dataframe, price array, tech array and time array to the specified data folder. _save_to_disk(data, file_path): save
the data to the specified file path

"""

import os
import pickle

from config_main import (
    TICKER_LIST,
    TIMEFRAME,
    no_candles_for_train,
    no_candles_for_val,
    TECHNICAL_INDICATORS_LIST,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    VAL_START_DATE,
    VAL_END_DATE
)
from processor_Binance import BinanceProcessor


def print_config_variables():
    print('\n')
    print('TIMEFRAME:                ', TIMEFRAME)
    print('no_candles_for_train:     ', no_candles_for_train)
    print('no_candles_for_val:       ', no_candles_for_val)
    print('TRAIN_START_DATE:         ', TRAIN_START_DATE)
    print('TRAIN_END_DATE:           ', TRAIN_END_DATE)
    print('VAL_START_DATE:           ', VAL_START_DATE)
    print('VAL_END_DATE:             ', VAL_END_DATE, '\n')
    print('TICKER LIST:              ', TICKER_LIST, '\n')


def process_data():
    DataProcessor = BinanceProcessor()
    data_from_processor, price_array, tech_array, time_array = DataProcessor.run(
        TICKER_LIST,
        TRAIN_START_DATE,
        VAL_END_DATE,
        TIMEFRAME,
        TECHNICAL_INDICATORS_LIST,
        if_vix=False
    )
    return data_from_processor, price_array, tech_array, time_array


def save_data(data_folder, data_from_processor, price_array, tech_array, time_array):
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    with open(data_folder + '/data_from_processor', 'wb') as handle:
        pickle.dump(data_from_processor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_folder + '/price_array', 'wb') as handle:
        pickle.dump(price_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_folder + '/tech_array', 'wb') as handle:
        pickle.dump(tech_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_folder + '/time_array', 'wb') as handle:
        pickle.dump(time_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_data_to_disk(data_from_processor, price_array, tech_array, time_array):
    data_folder = f'./data/{TIMEFRAME}_{no_candles_for_train + no_candles_for_val}'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    _save_to_disk(data_from_processor, f"{data_folder}/data_from_processor")
    _save_to_disk(price_array, f"{data_folder}/price_array")
    _save_to_disk(tech_array, f"{data_folder}/tech_array")
    _save_to_disk(time_array, f"{data_folder}/time_array")


def _save_to_disk(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    print_config_variables()
    data_from_processor, price_array, tech_array, time_array = process_data()
    save_data_to_disk(data_from_processor, price_array, tech_array, time_array)


if __name__ == "__main__":
    main()
