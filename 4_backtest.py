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
from processor_Binance import BinanceProcessor
from environment_Alpaca import CryptoEnvAlpaca
from environment_CCXT import CryptoEnvCCXT
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


def download_external_indicator(trade_start_date, trade_end_date):
    """
    Download external indicator data (CVIX alternative) from Binance or create dummy data.
    This function replaces CVOL-USD data which was previously fetched from Yahoo Finance.
    """
    try:
        # Try to get Bitcoin dominance index or another market indicator from Binance
        # For now, create dummy data that mimics volatility patterns

        # Create time range for the trading period
        start_date = pd.to_datetime(trade_start_date[:10])
        end_date = pd.to_datetime(trade_end_date[:10])
        time_range = pd.date_range(start=start_date, end=end_date, freq='5Min')

        # Generate realistic-looking indicator data (simulating volatility index)
        np.random.seed(42)  # For reproducibility
        base_value = 50.0

        # Create more realistic patterns
        values = []
        current_value = base_value

        for i in range(len(time_range)):
            # Random walk with mean reversion
            change = np.random.normal(0, 2)  # Small random changes
            mean_reversion = (base_value - current_value) * 0.01  # Gentle pull to mean

            current_value = current_value + change + mean_reversion
            current_value = np.clip(current_value, 20, 150)  # Keep in reasonable range

            values.append(current_value)

        indicator_series = pd.Series(values, index=time_range)
        indicator_series.name = 'close'

        print("Using dummy external indicator data (CVIX replacement)")
        return indicator_series.to_frame()

    except Exception as e:
        print(f"Error creating external indicator data: {e}")
        # Fallback: flat values
        time_range = pd.date_range(start=trade_start_date[:10],
                                  end=trade_end_date[:10],
                                  freq='5Min')
        dummy_series = pd.Series(50.0, index=time_range, name='close')
        return dummy_series.to_frame()


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

    # Load external indicator data (could be CVIX, volatility, etc.)
    external_data = download_external_indicator(trade_start_date, trade_end_date)
    time_series = time_array.to_series()
    time_series.name = 'time'
    external_data = pd.merge(time_series, external_data, left_index=True, right_index=True, how='left')
    indicator_array = external_data['close'].values
    indicator_array_growth = np.diff(indicator_array)

    return data_from_processor, price_array, tech_array, time_array, indicator_array, indicator_array_growth


# Inputs
#######################################################################################################
#######################################################################################################
#######################################################################################################

print('TRADE_START_DATE             ', trade_start_date)
print('TRADE_END_DATE               ', trade_end_date, '\n')

pickle_results = ["res_2025-12-07__01_28_01_model_CPCV_ppo_5m_50H_2k",
                  "res_2025-12-07__01_44_00_model_CPCV_ppo_5m_50H_2k",
                  "res_2025-12-07__10_33_02_model_KCV_ppo_5m_50H_2k",
                  "res_2025-12-07__14_51_43_model_WF_ppo_5m_50H_2k"
                  ]

# Execution
#######################################################################################################
#######################################################################################################

drl_cumrets_list = []
model_names_list = []

_, _, timeframe, ticker_list, technical_ind, _, _ = load_validated_model(pickle_results[0])
data_from_processor, price_array, tech_array, time_array, indicator_array, indicator_array_growth = load_and_process_data(TIMEFRAME, trade_start_date, trade_end_date)

for count, result in enumerate(pickle_results):
    env_params, net_dim, timeframe, ticker_list, technical_ind, name_test, model_name = load_validated_model(result)
    model_names_list.append(model_name)
    cwd = './train_results/' + result + '/stored_agent/'

    data_config = {
        "cvix_array": indicator_array,
        "cvix_array_growth": indicator_array_growth,
        "time_array": time_array,
        "price_array": price_array,
        "tech_array": tech_array,
        "if_train": False,
    }

    # Use CCXT environment for crypto trading
    env = CryptoEnvCCXT
    env_instance = env(config=data_config,
                       env_params=env_params,
                       if_log=True,
                       exchange_name='binance'
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

    # Slice indicator array
    if count == 0:
        indicator_array = indicator_array[indice_start:indice_end]
        indicator_array_growth = indicator_array_growth[indice_start:indice_end]

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
        # S&P index data removed - not available for Binance setup
        print("S&P Broad Crypto Market Index skipped - not available for Binance setup")

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
# S&P BDM Index removed - not available for Binance setup
ax1.plot(time_array, eqw_cumrets[1:], linewidth=line_width, label='Equal-weight', color='blue')


for i in range(np.shape(drl_rets_array)[1]):
    ax1.plot(time_array, drl_rets_array[:, i], label=model_names_list[i], linewidth=line_width)
ax1.legend(frameon=False, ncol=len(model_names_list) + 1, loc='upper left', bbox_to_anchor=(0, 1.11))
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth(3)
ax1.grid()

# Plot External Indicator
ax2 = ax1.twinx()
ax2.plot(time_array, indicator_array, linewidth=4, label='External Indicator', color='black', linestyle='dashed', alpha=0.4)
ax2.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.7, 1.17))
ax2.patch.set_edgecolor('black')
ax2.patch.set_linewidth(3)
ax2.set_ylabel('External Indicator')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=8))
ax1.set_ylabel('Cumulative return')
plt.xlabel('Date')
plt.legend()
plt.savefig('./plots_and_metrics/test_cumulative_return.png', bbox_inches='tight')
ax2.patch.set_edgecolor('black')
ax2.patch.set_linewidth(3)
ax2.set_ylabel('External Indicator')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=8))
ax1.set_ylabel('Cumulative return')
plt.xlabel('Date')
plt.legend()
plt.savefig('./plots_and_metrics/test_derivative_external_indicator.png', bbox_inches='tight')