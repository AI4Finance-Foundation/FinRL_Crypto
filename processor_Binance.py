"""

Reference: https://github.com/AI4Finance-LLC/FinRL

This code  defines a class BinanceProcessor which is used to process financial data from the Binance exchange. It
utilizes the Client class from the binance library to interact with the Binance API and the talib library to perform
technical analysis on the data.

The class has several methods, including __init__, run, download_data, clean_data, add_technical_indicator,
drop_correlated_features, add_vix, df_to_array, servertime_to_datetime, get_binance_bars,
and get_TALib_features_for_each_coin.

__init__ is the constructor for the class. It sets several instance variables and assigns values to them.

run method takes in several parameters, including ticker_list, start_date, end_date, time_interval,
technical_indicator_list, and if_vix. It sets the start_date and end_date to the instance variables and call several
other methods such as download_data, clean_data, add_technical_indicator, drop_correlated_features, add_vix,
and df_to_array to process the data.

download_data method takes in several parameters such as ticker_list, start_date, end_date, time_interval and calls
get_binance_bars method to download the data from Binance and returns the final dataframe.

clean_data method takes in a dataframe and drops any NaN values from it.

add_technical_indicator method takes in a dataframe and a list of technical indicators, and applies these indicators
to the dataframe and returns the updated dataframe.

drop_correlated_features method drops features that are highly correlated with other features.

add_vix method adds VIX data to the dataframe

df_to_array method converts dataframe to array

servertime_to_datetime converts timestamp to datetime

get_binance_bars method retrieves historical candlestick data from Binance.

get_TALib_features_for_each_coin method is used to calculate technical indicators for each coin using TALib library.

"""

import pandas as pd
from datetime import datetime
import numpy as np
from binance.client import Client
from talib import RSI, MACD, CCI, DX, ROC, ULTOSC, WILLR, OBV, HT_DCPHASE

from config_api import *
import datetime as dt
from processor_Yahoo import Yahoofinance
from fracdiff.sklearn import FracdiffStat

binance_client = Client(api_key=API_KEY_BINANCE, api_secret=API_SECRET_BINANCE)

class BinanceProcessor():
    def __init__(self):
        self.end_date = None
        self.start_date = None
        self.tech_indicator_list = None
        self.correlation_threshold = 0.9
        self.binance_api_key = API_KEY_BINANCE  # Enter your own API-key here
        self.binance_api_secret = API_SECRET_BINANCE  # Enter your own API-secret here
        self.binance_client = Client(api_key=API_KEY_BINANCE, api_secret=API_SECRET_BINANCE)

    def run(self, ticker_list, start_date, end_date, time_interval, technical_indicator_list, if_vix):
        self.start_date = start_date
        self.end_date = end_date
        print('Downloading data from Binance...')
        data = self.download_data(ticker_list, start_date, end_date, time_interval)
        print('Downloading finished! Transforming data...')
        data = self.clean_data(data)
        data = data.drop(columns=['time'])
        data['timestamp'] = self.servertime_to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
        data = self.add_technical_indicator(data, technical_indicator_list)
        data = self.drop_correlated_features(data)

        if if_vix:
            data = self.add_vix(data)

        price_array, tech_array, time_array = self.df_to_array(data, if_vix)
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        # Fracdiff input arrays
        # tech_array = self.frac_diff_features(tech_array)

        return data, price_array, tech_array, time_array

    # main functions
    def download_data(self, ticker_list, start_date, end_date,
                      time_interval):

        self.start_time = start_date
        self.end_time = end_date
        self.interval = time_interval
        self.ticker_list = ticker_list

        final_df = pd.DataFrame()
        for i in ticker_list:
            hist_data = self.get_binance_bars(self.start_time, self.end_time, self.interval, symbol=i)
            df = hist_data.iloc[:-1]
            df = df.dropna()
            df['tic'] = i
            final_df = final_df.append(df)

        return final_df

    def frac_diff_features(self, array):
        print('Differentiating tech array...')
        array = FracdiffStat().fit_transform(array)
        return array

    def clean_data(self, df):
        df = df.dropna()
        return df

    def add_technical_indicator(self, df, tech_indicator_list):
        final_df = pd.DataFrame()
        for i in df.tic.unique():
            # use massive function in previous cell
            coin_df = df[df.tic == i].copy()
            coin_df = self.get_TALib_features_for_each_coin(coin_df)

            # Append constructed tic_df
            final_df = final_df.append(coin_df)

        return final_df

    def drop_correlated_features(self, df):
        corr_matrix = pd.DataFrame(df).corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]

        to_drop.remove('close')
        print('according to analysis, drop: ', to_drop)
        real_drop = ['high', 'low', 'open', 'macd', 'cci', 'roc', 'willr']
        print('dropping for model consistency: ', real_drop)

        df_uncorrelated = df.drop(real_drop, axis=1)
        return df_uncorrelated

    def add_turbulence(self, df):
        print('Turbulence not supported yet. Return original DataFrame.')

        return df

    def add_5m_CVIX(self, df):
        trade_start_date = self.start_date[:10]
        trade_end_date = self.end_date[:10]
        TIME_INTERVAL = '60m'
        YahooProcessor = Yahoofinance('yahoofinance', trade_start_date, trade_end_date, TIME_INTERVAL)
        CVOL_df = YahooProcessor.download_data(['CVOL-USD'])
        CVOL_df.set_index('date', inplace=True)
        CVOL_df = CVOL_df.resample('5Min').interpolate(method='linear')
        df['CVIX'] = CVOL_df['close']
        return df

    def df_to_array(self, df, if_vix):
        self.tech_indicator_list = list(df.columns)
        self.tech_indicator_list.remove('tic')
        print('adding technical indiciators (no:', len(self.tech_indicator_list), ') :', self.tech_indicator_list)

        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][['close']].values
                tech_array = df[df.tic == tic][self.tech_indicator_list].values
                if_first_time = False
            else:
                price_array = np.hstack([price_array, df[df.tic == tic][['close']].values])
                tech_array = np.hstack([tech_array, df[df.tic == tic][self.tech_indicator_list].values])

            time_array = df[df.tic == self.ticker_list[0]].index

        assert price_array.shape[0] == tech_array.shape[0]

        return price_array, tech_array, time_array

    # helper functions
    def stringify_dates(self, date: datetime):
        return str(int(date.timestamp() * 1000))

    def servertime_to_datetime(self, timestamp):
        list_regular_stamps = [0] * len(timestamp)
        for indx, ts in enumerate(timestamp):
            list_regular_stamps[indx] = dt.datetime.fromtimestamp(ts / 1000)
        return list_regular_stamps

    def get_binance_bars(self, start_date, end_date, kline_size, symbol):
        data_df = pd.DataFrame()
        klines = self.binance_client.get_historical_klines(symbol, kline_size, start_date, end_date)
        data = pd.DataFrame(klines,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        data = data.drop(labels=['close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'], axis=1)
        if len(data_df) > 0:
            temp_df = pd.DataFrame(data)
            data_df = data_df.append(temp_df)
        else:
            data_df = data

        data_df = data_df.apply(pd.to_numeric, errors='coerce')
        data_df['time'] = [datetime.fromtimestamp(x / 1000.0) for x in data_df.timestamp]
        # data.drop(labels=["timestamp"], axis=1)
        data_df.index = [x for x in range(len(data_df))]

        return data_df

    def get_TALib_features_for_each_coin(self, tic_df):

        tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
        tic_df['macd'], _, _ = MACD(tic_df['close'], fastperiod=12,
                                    slowperiod=26, signalperiod=9)
        tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        tic_df['roc'] = ROC(tic_df['close'], timeperiod=10)
        tic_df['ultosc'] = ULTOSC(tic_df['high'], tic_df['low'], tic_df['close'])
        tic_df['willr'] = WILLR(tic_df['high'], tic_df['low'], tic_df['close'])
        tic_df['obv'] = OBV(tic_df['close'], tic_df['volume'])
        tic_df['ht_dcphase'] = HT_DCPHASE(tic_df['close'])

        return tic_df
