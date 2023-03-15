"""This code imports the necessary modules for processing and saving trade data, specifically os, pickle,
BinanceProcessor and config_main. The code also defines several functions for performing different tasks.

main(): This function is the entry point of the script. It calls the print_config_variables() function to print the
configuration variables, calls the process_data() function to process data, and then calls the save_data_to_disk()
function to save the processed data to disk.

print_config_variables(): This function prints the values of the configuration variables TIMEFRAME, TRADE_START_DATE,
TRADE_END_DATE, and TICKER_LIST to the console.

process_data(): This function creates an instance of the BinanceProcessor class, and then calls its run() function.
The function passes in the configuration variables TICKER_LIST, trade_start_date, trade_end_date, TIMEFRAME,
and TECHNICAL_INDICATORS_LIST as arguments to the run() function. The function returns the result of the run()
function, which is a tuple of four arrays: data_from_processor, price_array, tech_array, and time_array.

save_data_to_disk(): This function creates a folder called data/trade_data/{TIMEFRAME}_{no_candles_for_train} if it
doesn't exist. Then it saves the four arrays from the process_data() function to disk using the _save_to_disk()
function.

_save_to_disk(): This is a helper function that saves an array to disk using the pickle module. The function takes in
two arguments, data and file_path, and saves the data to the specified file path.

The code also includes an if __name__ == "__main__": block at the end, which calls the main() function when the
script is run. This ensures that the script only runs when it is executed directly and not when it is imported as a
module."""


import os
import pickle

from processor_Binance import BinanceProcessor
from config_main import TICKER_LIST, TECHNICAL_INDICATORS_LIST, TIMEFRAME, trade_start_date, trade_end_date, no_candles_for_train


def main():
    print_config_variables()
    data_from_processor, price_array, tech_array, time_array = process_data()
    save_data_to_disk(data_from_processor, price_array, tech_array, time_array)


def print_config_variables():
    print('\n')
    print('TIMEFRAME                  ', TIMEFRAME)
    print('TRADE_START_DATE           ', trade_start_date)
    print('TRADE_END_DATA             ', trade_end_date)
    print('TICKER LIST                ', TICKER_LIST, '\n')


def process_data():
    data_processor = BinanceProcessor()
    return data_processor.run(TICKER_LIST, trade_start_date, trade_end_date, TIMEFRAME, TECHNICAL_INDICATORS_LIST,
                              if_vix=False)


def save_data_to_disk(data_from_processor, price_array, tech_array, time_array):
    data_folder = f'./data/trade_data/{TIMEFRAME}_{str(trade_start_date[2:10])}_{str(trade_end_date[2:10])}'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    _save_to_disk(data_from_processor, f"{data_folder}/data_from_processor")
    _save_to_disk(price_array, f"{data_folder}/price_array")
    _save_to_disk(tech_array, f"{data_folder}/tech_array")
    _save_to_disk(time_array, f"{data_folder}/time_array")


def _save_to_disk(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
