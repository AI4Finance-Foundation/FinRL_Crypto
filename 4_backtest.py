"""
This script contains a set of functions for loading and processing data for trading.

It contains the following functions:

load_validated_model: Loads the best trial from the pickle file in the specified directory and returns the best trial's attributes.
download_CVIX: Downloads the CVIX dataframe from Yahoo finance and returns it.
load_and_process_data: loads and process the trade data from the specified data folder and returns the data.

After that, the large loop analyzes every result by creating an instance of an Alpaca environment and checking
what the model would do through the environment using the new trading data

Finally, the resulting backtests are analyzes for performance a performance metric per benchmark (EQW, S&P BCI) plus
all the input DRL agents are analyzed.

"""


import pickle
import matplotlib.dates as mdates

from config_main import *
from function_finance_metrics import *
from processor_Yahoo import Yahoofinance
from environment_Alpaca import CryptoEnvAlpaca
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl


def load_validated_model(result):

    with open('./train_results/' + result + '/best_trial', 'rb') as handle:
        best_trial = pickle.load(handle)

    print('BEST TRIAL: ', best_trial.number)
    timeframe = best_trial.user_attrs['timeframe']
    ticker_list = best_trial.user_attrs['ticker_list']
    technical_ind = best_trial.user_attrs['technical_indicator_list']
    net_dim = best_trial.params['net_dimension']
    model_name = best_trial.user_attrs['model_name']

    print('\nMODEL_NAME: ', model_name)
    print(best_trial.params)
    print(timeframe)

    name_test = best_trial.user_attrs['name_test']

    env_params = {
        "lookback": best_trial.params['lookback'],
        "norm_cash": best_trial.params['norm_cash'],
        "norm_stocks": best_trial.params['norm_stocks'],
        "norm_tech": best_trial.params['norm_tech'],
        "norm_reward": best_trial.params['norm_reward'],
        "norm_action": best_trial.params['norm_action']
    }
    return env_params, net_dim, timeframe, ticker_list, technical_ind, name_test, model_name


def download_CVIX(trade_start_date, trade_end_date):
    trade_start_date = trade_start_date[:10]
    trade_end_date = trade_end_date[:10]
    TIME_INTERVAL = '60m'
    YahooProcessor = Yahoofinance('yahoofinance', trade_start_date, trade_end_date, TIME_INTERVAL)
    CVOL_df = YahooProcessor.download_data(['CVOL-USD'])
    CVOL_df.set_index('date', inplace=True)
    CVOL_df = CVOL_df.resample('5Min').interpolate(method='linear')
    return CVOL_df['close']


def load_and_process_data(TIMEFRAME, trade_start_date, trade_end_date):
    data_folder = f'./data/trade_data/{TIMEFRAME}_{str(trade_start_date[2:10])}_{str(trade_end_date[2:10])}'
    print(f'\nLOADING DATA FOLDER: {data_folder}\n')
    with open(data_folder + '/data_from_processor', 'rb') as handle:
        data_from_processor = pickle.load(handle)
    with open(data_folder + '/price_array', 'rb') as handle:
        price_array = pickle.load(handle)
    with open(data_folder + '/tech_array', 'rb') as handle:
        tech_array = pickle.load(handle)
    with open(data_folder + '/time_array', 'rb') as handle:
        time_array = pickle.load(handle)

    CVIX_df = download_CVIX(trade_start_date, trade_end_date)
    CVIX_df = pd.merge(time_array.to_series(), CVIX_df, left_index=True, right_index=True, how='left')
    cvix_array = CVIX_df['close'].values
    cvix_array_growth = np.diff(cvix_array)

    return data_from_processor, price_array, tech_array, time_array, cvix_array, cvix_array_growth


# Inputs
#######################################################################################################
#######################################################################################################
#######################################################################################################

print('TRADE_START_DATE             ', trade_start_date)
print('TRADE_END_DATE               ', trade_end_date, '\n')

pickle_results = ["res_2023-01-23__16_32_55_model_WF_ppo_5m_3H_20k",
                  "res_2023-01-23__17_07_49_model_KCV_ppo_5m_3H_20005k",
                  "res_2023-01-23__16_44_30_model_CPCV_ppo_5m_3H_20k"
                  ]

# Execution
#######################################################################################################
#######################################################################################################

drl_cumrets_list = []
model_names_list = []

_, _, timeframe, ticker_list, technical_ind, _, _ = load_validated_model(pickle_results[0])
data_from_processor, price_array, tech_array, time_array, cvix_array, cvix_array_growth = load_and_process_data(TIMEFRAME, trade_start_date, trade_end_date)

for count, result in enumerate(pickle_results):
    env_params, net_dim, timeframe, ticker_list, technical_ind, name_test, model_name = load_validated_model(result)
    model_names_list.append(model_name)
    cwd = './train_results/' + result + '/stored_agent/'

    data_config = {
        "cvix_array": cvix_array,
        "cvix_array_growth": cvix_array_growth,
        "time_array": time_array,
        "price_array": price_array,
        "tech_array": tech_array,
        "if_train": False,
    }

    env = CryptoEnvAlpaca
    env_instance = env(config=data_config,
                       env_params=env_params,
                       if_log=True
                       )

    account_value_erl = DRLAgent_erl.DRL_prediction(
        model_name=model_name,
        cwd=cwd,
        net_dimension=net_dim,
        environment=env_instance,
        gpu_id=0
    )

    # Correct slicing (due to DRL start/end)
    lookback = env_params['lookback']
    indice_start = lookback - 1
    indice_end = len(price_array) - lookback
    time_array = time_array[indice_start:indice_end]

    # Slice cvix array
    if count == 0:
        cvix_array = cvix_array[indice_start:indice_end]
        cvix_array_growth = cvix_array_growth[indice_start:indice_end]

    # Compute Sharpe's of each coin
    account_value_eqw, ewq_rets, eqw_cumrets = compute_eqw(price_array, indice_start, indice_end)

    # Compute annualization factor
    data_points_per_year = compute_data_points_per_year(timeframe)
    dataset_size = np.shape(ewq_rets)[0]
    factor = data_points_per_year / dataset_size

    # Compute DRL rets
    account_value_erl = np.array(account_value_erl)
    drl_rets = account_value_erl[1:] - account_value_erl[:-1]
    drl_cumrets = [x / account_value_erl[0] - 1 for x in account_value_erl]
    drl_cumrets_list.append(drl_cumrets)

    # Compute metrics per pickle result
    #######################################################################################################

    # Only compute consistent metrics once
    if count == 0:
        # Load S&P index
        spy_index_df = pd.read_csv('data/SPY_Crypto_Broad_Digital_Market_Index - Sheet1.csv')
        spy_index_df['Date'] = pd.to_datetime(spy_index_df['Date'])

        account_value_spy = np.array(spy_index_df['S&P index'])
        spy_rets = account_value_spy[:-1] / account_value_spy[1:] - 1
        spy_rets = np.insert(spy_rets, 0, 0)
        spy_index_df['cumrets_sp_idx'] = [x / spy_index_df['S&P index'][0] - 1 for x in spy_index_df['S&P index']]
        spy_index_df['rets_sp_idx'] = spy_rets
        spy_index_df.set_index('Date', inplace=True)
        spy_index_df = spy_index_df.resample('5Min').interpolate(method='pchip')

        sp_annual_ret, sp_annual_vol, sp_sharpe_rat, sp_vol = aggregate_performance_array(spy_rets, factor)

        write_metrics_to_results('S&P Broad Crypto index',
                                 'plots_and_metrics/test_metrics.txt',
                                 spy_index_df['cumrets_sp_idx'],
                                 sp_annual_ret,
                                 sp_annual_vol,
                                 sp_sharpe_rat,
                                 sp_vol,
                                 'w'
                                 )

        # Write buy-and-hold strategy
        eqw_annual_ret, eqw_annual_vol, eqw_sharpe_rat, eqw_vol = aggregate_performance_array(np.array(ewq_rets),
                                                                                                 factor)
        write_metrics_to_results('Buy-and-Hold',
                                 'plots_and_metrics/test_metrics.txt',
                                 eqw_cumrets,
                                 eqw_annual_ret,
                                 eqw_annual_vol,
                                 eqw_sharpe_rat,
                                 eqw_vol,
                                 'a'
                                 )

    # Then compute the actual metrics from the DRL agents
    drl_annual_ret, drl_annual_vol, drl_sharpe_rat, drl_vol = aggregate_performance_array(np.array(drl_rets), factor)
    write_metrics_to_results(model_name,
                             'plots_and_metrics/test_metrics.txt',
                             drl_cumrets,
                             drl_annual_ret,
                             drl_annual_vol,
                             drl_sharpe_rat,
                             drl_vol,
                             'a'
                             )

    # Hold out of loop only add once
    #######################################################################################################

# Plot
#######################################################################################################
#######################################################################################################

drl_rets_array = np.transpose(np.vstack(drl_cumrets_list))

# General 1
plt.rcParams.update({'font.size': 22})
plt.figure(dpi=300)
f, ax1 = plt.subplots(figsize=(20, 8))

# Plot returns
line_width = 2
ax1.plot(spy_index_df.index, spy_index_df['cumrets_sp_idx'].values, linewidth=3, label='S&P BDM Index')
ax1.plot(time_array, eqw_cumrets[1:], linewidth=line_width, label='Equal-weight', color='blue')


for i in range(np.shape(drl_rets_array)[1]):
    ax1.plot(time_array, drl_rets_array[:, i], label=model_names_list[i], linewidth=line_width)
ax1.legend(frameon=False, ncol=len(model_names_list) + 2, loc='upper left', bbox_to_anchor=(0, 1.11))
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth(3)
ax1.grid()

# Plot CVIX
ax2 = ax1.twinx()
ax2.plot(time_array, cvix_array, linewidth=4, label='CVIX', color='black', linestyle='dashed', alpha=0.4)
ax2.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.7, 1.17))
ax2.patch.set_edgecolor('black')
ax2.patch.set_linewidth(3)
ax2.set_ylabel('CVIX')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=8))
ax1.set_ylabel('Cumulative return')
plt.xlabel('Date')
plt.legend()
plt.savefig('./plots_and_metrics/test_cumulative_return.png', bbox_inches='tight')
ax2.patch.set_edgecolor('black')
ax2.patch.set_linewidth(3)
ax2.set_ylabel('CVIX')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=8))
ax1.set_ylabel('Cumulative return')
plt.xlabel('Date')
plt.legend()
plt.savefig('./plots_and_metrics/test_derivative_CVIX.png', bbox_inches='tight')