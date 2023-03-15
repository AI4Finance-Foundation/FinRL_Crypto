"""
Reference: https://github.com/AI4Finance-LLC/FinRL'''

The code above defines a class Yahoofinance which is used to process financial data from the Yahoo Finance API using
the yfinance library. It also utilizes numpy, pandas and pytz libraries. The class inherits from _Base class which is
imported from processor_Base.

The class has several methods, including __init__, download_data, clean_data, get_trading_days,
add_technical_indicator, df_to_array and servertime_to_datetime.

__init__ is the constructor for the class. It calls the superclass' __init__ method to set the data_source,
start_date, end_date and time_interval.

download_data method takes in a list of ticker symbols and uses the yfinance library to download the data for each
ticker symbol, formats the data and returns it as a DataFrame.

clean_data method takes the dataframe and cleans the data. It renames columns, gets the trading days,
fills any missing data and sorts the dataframe by date and ticker.

get_trading_days method takes in start and end date and returns the trading days between these dates.

add_technical_indicator method adds technical indicators to the dataframe.

df_to_array converts dataframe to array

servertime_to_datetime converts timestamp to datetime

It is also worth noting that, the class uses exchange_calendars library to get trading days, but if it fails to
import it, it uses trading_calendars library instead.

"""

from typing import List
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from datetime import datetime
try:
    import exchange_calendars as tc
except:
    print('Cannot import exchange_calendars.', 
          'If you are using python>=3.7, please install it.')
    import trading_calendars as tc
    print('Use trading_calendars instead for yahoofinance processor..')
# from basic_processor import _Base
from processor_Base import _Base


class Yahoofinance(_Base):
    def __init__(self, data_source: str, start_date: str, end_date: str, time_interval: str, **kwargs):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)

    def download_data(self, ticker_list: List[str]):
        #self.time_zone = calc_time_zone(ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED)

        # Download and save the data in a pandas DataFrame:
        self.dataframe = pd.DataFrame()
        for tic in ticker_list:
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date, interval=self.time_interval)
            temp_df["tic"] = tic
            self.dataframe = self.dataframe.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
        self.dataframe.reset_index(inplace=True)
        try:
            # convert the column names to standardized names
            self.dataframe.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
                "tic",
            ]
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        # self.dataframe["day"] = self.dataframe["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        self.dataframe["date"] = self.dataframe.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        self.dataframe["date"] = self.dataframe.date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))


        # drop missing data
        self.dataframe.dropna(inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        print("Shape of DataFrame: ", self.dataframe.shape)
        # print("Display DataFrame: ", data_df.head())

        self.dataframe.sort_values(by=['date', 'tic'], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        return self.dataframe

    def clean_data(self):

        df = self.dataframe.copy()
        df = df.rename(columns={'date': 'time'})
        time_interval = self.time_interval
        # get ticker list
        tic_list = np.unique(df.tic.values)

        # get complete time index
        trading_days = self.get_trading_days(start=self.start_date, end=self.end_date)
        if time_interval == '1D':
            times = trading_days
        elif time_interval == '1Min':
            times = []
            for day in trading_days:
                current_time = pd.Timestamp(day + ' 09:30:00').tz_localize(self.time_zone)
                for _ in range(390):
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=1)
        else:
            raise ValueError('Data clean at given time interval is not supported for YahooFinance data.')

        # fill NaN data
        new_df = pd.DataFrame()
        for tic in tic_list:
            print(('Clean data for ') + tic)
            # create empty DataFrame using complete time index
            tmp_df = pd.DataFrame(columns=['open', 'high', 'low', 'close',
                                           'adjusted_close', 'volume'],
                                  index=times)
            # get data for current ticker
            tic_df = df[df.tic == tic]
            # fill empty DataFrame using orginal data
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]['time']] = tic_df.iloc[i] \
                    [['open', 'high', 'low', 'close', 'adjusted_close', 'volume']]

            # if close on start date is NaN, fill data with first valid close
            # and set volume to 0.
            if str(tmp_df.iloc[0]['close']) == 'nan':
                print('NaN data on start date, fill using first valid data.')
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]['close']) != 'nan':
                        first_valid_close = tmp_df.iloc[i]['close']
                        first_valid_adjclose = tmp_df.iloc[i]['adjusted_close']

                tmp_df.iloc[0] = [first_valid_close, first_valid_close,
                                  first_valid_close, first_valid_close,
                                  first_valid_adjclose, 0.0]

            # fill NaN data with previous close and set volume to 0.
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]['close']) == 'nan':
                    previous_close = tmp_df.iloc[i - 1]['close']
                    previous_adjusted_close = tmp_df.iloc[i - 1]['adjusted_close']
                    if str(previous_close) == 'nan':
                        raise ValueError
                    tmp_df.iloc[i] = [previous_close, previous_close, previous_close,
                                      previous_close, previous_adjusted_close, 0.0]

            # merge single ticker data to new DataFrame
            tmp_df = tmp_df.astype(float)
            tmp_df['tic'] = tic
            new_df = new_df.append(tmp_df)

            print(('Data clean for ') + tic + (' is finished.'))

        # reset index and rename columns
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={'index': 'time'})

        print('Data clean all finished!')

        self.dataframe = new_df

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar('NYSE')
        df = nyse.sessions_in_range(pd.Timestamp(start, tz=pytz.UTC),
                                    pd.Timestamp(end, tz=pytz.UTC))
        return [str(day)[:10] for day in df]
