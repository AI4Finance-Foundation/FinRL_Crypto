"""
This code defines various settings and functions for training a machine learning model.

The trade start date in combinations with the amount of candles required for training and validation determine all
other parameters automatically.

The function nCr calculates the number of ways to choose r elements from a set of n elements, also known as a combination.

The settings defined in this script include the random seed SEED_CFG, the time frame TIMEFRAME,
the number of trials H_TRIALS,
the number of groups used for testing K_TEST_GROUPS,
the number of paths NUM_PATHS,
the number of K-fold cross validation groups KCV_groups
the number of groups N_GROUPS,
the number of splits NUMBER_OF_SPLITS,
the start and end date for the trade period trade_start_date
the trade_end_date,
the number of candles for training no_candles_for_train
the validation no_candles_for_val
the list of tickers TICKER_LIST,
the minimum buy limits ALPACA_LIMITS,
the list of technical indicators TECHNICAL_INDICATORS_LIST.

The function calculate_start_end_dates is used to compute the start and end dates for training and validation based on the number of candles and the selected time frame.

"""

from datetime import datetime, timedelta
import numpy as np
import operator as op
from functools import reduce


def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


# General Training Settings
#######################################################################################################
#######################################################################################################

trade_start_date = '2022-04-30 00:00:00'
trade_end_date = '2022-06-27 00:00:00'

SEED_CFG = 2390408
TIMEFRAME = '5m'
H_TRIALS = 50
KCV_groups = 5
K_TEST_GROUPS = 2
NUM_PATHS = 4
N_GROUPS = NUM_PATHS + 1
NUMBER_OF_SPLITS = nCr(N_GROUPS, N_GROUPS - K_TEST_GROUPS)

print(NUMBER_OF_SPLITS)

no_candles_for_train = 20000
no_candles_for_val = 5000

TICKER_LIST = ['AAVEUSDT',
               'AVAXUSDT',
               'BTCUSDT',
               'NEARUSDT',
               'LINKUSDT',
               'ETHUSDT',
               'LTCUSDT',
               'MATICUSDT',
               'UNIUSDT',
               'SOLUSDT',
               ]


# Minimum buy limits
ALPACA_LIMITS = np.array([0.01,
                          0.10,
                          0.0001,
                          0.1,
                          0.1,
                          0.001,
                          0.01,
                          10,
                          0.1,
                          0.01
                          ])


TECHNICAL_INDICATORS_LIST = ['open',
                             'high',
                             'low',
                             'close',
                             'volume',
                             'macd',
                             'macd_signal',
                             'macd_hist',
                             'rsi',
                             'cci',
                             'dx'
                             ]


# Auto compute all necessary dates based on candle distribution
#######################################################################################################
#######################################################################################################

def calculate_start_end_dates(candlewidth):
    no_minutes = int

    candle_to_no_minutes = {'1m': 1, '5m': 5, '10m': 10, '30m': 30, '1h': 60, '2h': 2*60, '4h': 4*60, '12h': 12*60}
    no_minutes = candle_to_no_minutes[candlewidth]

    trade_start_date_datetimeObj = datetime.strptime(trade_start_date, "%Y-%m-%d %H:%M:%S")

    # train start date = trade_start_date - (no_c_t  + no_c_v)
    train_start_date = (trade_start_date_datetimeObj
                        - timedelta(minutes=no_minutes * (no_candles_for_train
                                                          + no_candles_for_val))).strftime("%Y-%m-%d %H:%M:%S")

    # train start date = trade_start_date - (no_c_v + 1)
    train_end_date = (trade_start_date_datetimeObj
                      - timedelta(minutes=no_minutes * (no_candles_for_val + 1))).strftime("%Y-%m-%d %H:%M:%S")

    # validation start date = trade_start_date - no_c_v
    val_start_date = (trade_start_date_datetimeObj
                      - timedelta(minutes=no_minutes * no_candles_for_val)).strftime("%Y-%m-%d %H:%M:%S")

    # validation start date = trade_start_date - 1
    val_end_date = (trade_start_date_datetimeObj
                    - timedelta(minutes=no_minutes * 1)).strftime("%Y-%m-%d %H:%M:%S")

    return train_start_date, train_end_date, val_start_date, val_end_date


TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE = calculate_start_end_dates(TIMEFRAME)
print("TRAIN_START_DATE: ", TRAIN_START_DATE)
print("VAL_END_DATE: ", VAL_END_DATE)
