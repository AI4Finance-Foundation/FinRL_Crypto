"""This code is for training a reinforcement learning agent for trading cryptocurrencies using Optuna and Alpaca API.
It includes functions for setting up the environment, cross-validation, and training, as well as saving the best
performing agent.

print_config(): This function prints out the current configuration settings for the training.

set_Pandas_Timedelta(TIMEFRAME): This is a helper function that converts the time frame input to a pandas Timedelta
object.

save_best_agent(study, trial): This is a callback function that is called at the end of each trial, it checks if the
current trial is the best one, and if so, it copies the agent from the working directory to the results directory and
pickles the trial object.

sample_hyperparams(trial): This function samples the hyperparameters for the agent during the training process.

objective(trial, name_test, model_name, cwd, res_timestamp, gpu_id) : This function is used as the objective function
for the Optuna optimization. It sets up the environment and cross-validation for the agent training and returns the
final evaluation metric as the objective value.

optimize(name_test:str, model_name:str, gpu_id:str) : This function optimizes the agent training using Optuna by
calling the objective function and specifying the search space for the hyperparameters. It also includes a callback
function to save the best agent and logging of the results.

setup_CPCV(erl_params, tech_array, time_array, NUM_PATHS, K_TEST_GROUPS, TIMEFRAME): This function sets up the Purged
Combinatorial Cross-Validation for the agent training. It takes in parameters for the environment, technical
indicators, and time frame, and returns the cross-validation object to be used in the training.

back_test_paths_generator(data, y, cv, n_samples, n_total_groups, k_test_groups, prediction_times, evaluation_times,
verbose): This function generates the paths for backtesting of the agent. It takes in the data, labels,
cross-validation object, number of samples, and parameters for the paths and returns the is_test, paths,
and evaluation times.

class bcolors: : This class defines the color codes for the terminal output.
"""

import joblib
import optuna
import datetime
import pickle
import os
import sys

from distutils.dir_util import copy_tree
from environment_Alpaca import CryptoEnvAlpaca
from function_CPCV import *
from function_train_test import *
from config_main import *


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_config():
    print('\n' + bcolors.HEADER + '##### Launched hyperparameter optimization with CPCV  #####' + bcolors.ENDC + '\n')
    print('TIMEFRAME                  ', TIMEFRAME)
    print('TRAIN SAMPLES              ', no_candles_for_train)
    print('TRIALS NO.                 ', H_TRIALS)
    print('N                          ', N_GROUPS)
    print('K test groups              ', K_TEST_GROUPS)
    print('SPLITS                     ', NUMBER_OF_SPLITS)

    print('\n')
    print('TRAIN SAMPLES              ', no_candles_for_train)
    print('VAL_SAMPLES                ', no_candles_for_val)
    print('TRAIN_START_DATE           ', TRAIN_START_DATE)
    print('TRAIN_END_DATE             ', TRAIN_END_DATE)
    print('VAL_START_DATE             ', VAL_START_DATE)
    print('VAL_END_DATE               ', VAL_END_DATE, '\n')
    print('TICKER LIST                ', TICKER_LIST, '\n')
    res_timestamp = 'res_' + str(datetime.now().strftime("%Y-%m-%d__%H_%M_%S"))
    return res_timestamp


def set_Pandas_Timedelta(TIMEFRAME):
    timeframe_to_delta = {'1m': pd.Timedelta(minutes=1),
                          '5m': pd.Timedelta(minutes=5),
                          '10m': pd.Timedelta(minutes=10),
                          '30m': pd.Timedelta(minutes=30),
                          '1h': pd.Timedelta(hours=1),
                          '1d': pd.Timedelta(days=1),
                          }
    if TIMEFRAME in timeframe_to_delta:
        return timeframe_to_delta[TIMEFRAME]
    else:
        raise ValueError('Timeframe not supported yet, please manually add!')


def save_best_agent(study, trial):
    if study.best_trial.number != trial.number:
        return

    print('\n' + bcolors.OKGREEN + 'Found new best agent!' + bcolors.ENDC + '\n')

    # Copy agent from workdir and save in result folder
    name_folder = trial.user_attrs['name_folder']
    name_test = trial.user_attrs['name_test']
    from_directory = f"./train_results/cwd_tests/{name_test}/"
    to_directory = f"./train_results/{name_folder}/stored_agent/"

    os.makedirs(to_directory, exist_ok=True)
    copy_tree(from_directory, to_directory)

    # Dump trial in pickle file to avoid error where params arre not copied
    with open(f"./train_results/{name_folder}/best_trial", "wb") as handle:
        pickle.dump(trial, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sample_hyperparams(trial):
    average_episode_step_min = no_candles_for_train + 0.25 * no_candles_for_train
    sampled_erl_params = {
        "learning_rate": trial.suggest_categorical("learning_rate", [3e-2, 2.3e-2, 1.5e-2, 7.5e-3, 5e-6]),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1280, 2048, 3080]),
        "gamma": trial.suggest_categorical("gamma", [0.85, 0.99, 0.999]),
        "net_dimension": trial.suggest_categorical("net_dimension", [2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12]),
        "target_step": trial.suggest_categorical("target_step",
                                                 [average_episode_step_min, round(1.5 * average_episode_step_min),
                                                  2 * average_episode_step_min]),
        "eval_time_gap": trial.suggest_categorical("eval_time_gap", [60]),
        "break_step": trial.suggest_categorical("break_step", [3e4, 4.5e4, 6e4])
    }

    # environment normalization and lookback
    sampled_env_params = {
        "lookback": trial.suggest_categorical("lookback", [1]),
        "norm_cash": trial.suggest_categorical("norm_cash", [2 ** -12]),
        "norm_stocks": trial.suggest_categorical("norm_stocks", [2 ** -8]),
        "norm_tech": trial.suggest_categorical("norm_tech", [2 ** -15]),
        "norm_reward": trial.suggest_categorical("norm_reward", [2 ** -10]),
        "norm_action": trial.suggest_categorical("norm_action", [10000])
    }
    return sampled_erl_params, sampled_env_params


def set_pickle_attributes(trial, model_name, TIMEFRAME, TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE, TICKER_LIST, TECHNICAL_INDICATORS_LIST, name_folder, name_test, study):
    # user attributes for saving in the pickle model file later
    trial.set_user_attr("model_name", model_name)
    trial.set_user_attr("timeframe", TIMEFRAME)
    trial.set_user_attr("train_start_date", TRAIN_START_DATE)
    trial.set_user_attr("train_end_date", TRAIN_END_DATE)
    trial.set_user_attr("test_start_date", VAL_START_DATE)
    trial.set_user_attr("test_end_date", VAL_END_DATE)
    trial.set_user_attr("ticker_list", TICKER_LIST)
    trial.set_user_attr("technical_indicator_list", TECHNICAL_INDICATORS_LIST)
    trial.set_user_attr("name_folder", name_folder)
    trial.set_user_attr("name_test", name_test)
    joblib.dump(study, f'train_results/{name_folder}/' + 'study.pkl')


def load_saved_data(TIMEFRAME, no_candles_for_train):
    data_folder = './data/' + TIMEFRAME + '_' + str(no_candles_for_train + no_candles_for_val)
    print('\nLOADING DATA FOLDER: ', data_folder, '\n')
    with open(data_folder + '/data_from_processor', 'rb') as handle:
        data_from_processor = pickle.load(handle)
    with open(data_folder + '/price_array', 'rb') as handle:
        price_array = pickle.load(handle)
    with open(data_folder + '/tech_array', 'rb') as handle:
        tech_array = pickle.load(handle)
    with open(data_folder + '/time_array', 'rb') as handle:
        time_array = pickle.load(handle)
    return data_from_processor, price_array, tech_array, time_array


def write_logs(name_folder, model_name, trial, cwd, erl_params, env_params, num_paths, n_total_groups, n_splits):
    path_logs = './train_results/' + name_folder + '/logs.txt'
    with open(path_logs, 'a') as f:
        f.write('\n' + 'MODEL NAME: ' + model_name + '\n')
        f.write('TRIAL NUMBER: ' + str(trial.number) + '\n')
        f.write('CWD: ' + cwd + '\n')
        f.write(str(erl_params) + '\n')
        f.write(str(env_params) + '\n')
        f.write('\n' + 'TIME START OUTER: ' + str(datetime.now()) + '\n')

        f.write('\n######### CPCV Settings #########' + '\n')
        f.write("Paths  : " + str(num_paths) + '\n')
        f.write("N      : " + str(n_total_groups) + '\n')
        f.write("splits : " + str(n_splits) + '\n\n')
    return path_logs


def setup_CPCV(data_from_processor, erl_params, tech_array, time_array, NUM_PATHS, K_TEST_GROUPS, TIMEFRAME):
    # Set constants
    env = CryptoEnvAlpaca
    break_step = erl_params["break_step"]

    # Setup Purged CombinatorialCross-Validation
    num_paths = NUM_PATHS
    k_test_groups = K_TEST_GROUPS
    n_total_groups = num_paths + 1
    t_final = 10
    embargo_td = set_Pandas_Timedelta(TIMEFRAME) * t_final * 5
    n_splits = np.array(list(itt.combinations(np.arange(n_total_groups), k_test_groups))).reshape(-1, k_test_groups)
    n_splits = len(n_splits)
    cv = CombPurgedKFoldCV(n_splits=n_total_groups, n_test_splits=k_test_groups, embargo_td=embargo_td)

    # Set placeholder target variable
    data = pd.DataFrame(tech_array)
    data = data.set_index(time_array)
    data.drop(data.tail(t_final).index, inplace=True)
    y = pd.Series([0] * data_from_processor.shape[0])
    y = y.reindex(data.index)
    y = y.squeeze()

    # prediction and evaluation times
    prediction_times = pd.Series(data.index, index=data.index)
    evaluation_times = pd.Series(data.index, index=data.index)

    # Compute paths
    is_test, paths, _ = back_test_paths_generator(data, y, cv, data.shape[0], n_total_groups, k_test_groups,
                                                  prediction_times, evaluation_times, verbose=False)

    return cv, env, data, y, num_paths, paths, n_total_groups, n_splits, break_step, prediction_times, evaluation_times


def objective(trial, name_test, model_name, cwd, res_timestamp, gpu_id):

    # Set full name_folder
    name_folder = res_timestamp + '_' + name_test

    set_pickle_attributes(trial, model_name, TIMEFRAME, TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE,
                          TICKER_LIST, TECHNICAL_INDICATORS_LIST, name_folder, name_test, study)

    # Sample set of hyperparameters
    erl_params, env_params = sample_hyperparams(trial)

    # Load data from hard disk
    data_from_processor, price_array, tech_array, time_array = load_saved_data(TIMEFRAME, no_candles_for_train)

    # Setup Combinatorial Purged Cross-Validation
    cpcv, \
        env, \
        data, y, \
        num_paths, \
        paths, \
        n_total_groups, \
        n_splits, \
        break_step, \
        prediction_times, \
        evaluation_times = setup_CPCV(data_from_processor, erl_params, tech_array, time_array, NUM_PATHS, K_TEST_GROUPS,
                                      TIMEFRAME)

    # initiate logs for tracking behaviour during training
    path_logs = write_logs(name_folder, model_name, trial, cwd, erl_params, env_params, num_paths, n_total_groups,
                           n_splits)

    # CPCV Split function eval
    #######################################################################################################
    #######################################################################################################

    # CV loop
    sharpe_list_bot = []
    sharpe_list_ewq = []
    drl_rets_val_list = []

    for split, (train_indices, test_indices) in enumerate(
            cpcv.split(data, y, pred_times=prediction_times, eval_times=evaluation_times)):

        with open(path_logs, 'a') as f:
            f.write('TIME START INNER: ' + str(datetime.now()))

        sharpe_bot, sharpe_eqw, drl_rets_tmp = train_and_test(trial, price_array, tech_array, train_indices,
                                                              test_indices, env, model_name, env_params,
                                                              erl_params, break_step, cwd, gpu_id)

        sharpe_list_ewq.append(sharpe_eqw)
        sharpe_list_bot.append(sharpe_bot)

        with open(path_logs, 'a') as f:
            f.write('\n' + 'SPLIT: ' + str(split) + '     # Optimizing for Sharpe ratio!' + '\n')
            f.write('BOT:         ' + str(sharpe_bot) + '\n')
            f.write('HODL:        ' + str(sharpe_eqw) + '\n')
            f.write('TIME END INNER: ' + str(datetime.now()) + '\n\n')

        # Fill the backtesting prediction matrix
        drl_rets_val_list.append(drl_rets_tmp)
        trial.set_user_attr("price_array", price_array)
        trial.set_user_attr("tech_array", tech_array)
        trial.set_user_attr("time_array", time_array)

    # Hyperparameter bjective function eval
    #######################################################################################################
    #######################################################################################################

    # Matrices
    trial.set_user_attr("drl_rets_val_list", drl_rets_val_list)

    # Interesting values
    trial.set_user_attr("sharpe_list_bot", sharpe_list_bot)
    trial.set_user_attr("sharpe_list_ewq", sharpe_list_ewq)
    trial.set_user_attr("paths", paths)

    with open(path_logs, 'a') as f:
        f.write('\nHYPERPARAMETER EVAL || SHARPE AVG BOT    :  ' + str(np.mean(sharpe_list_bot)) + '\n')
        f.write('HYPERPARAMETER EVAL || SHARPE AVG HODL     : ' + str(np.mean(sharpe_list_ewq)) + '\n')
        f.write('DIFFERENCE                                 : ' + str(
            np.mean(sharpe_list_bot) - np.mean(sharpe_list_ewq)) + '\n')
        f.write('\n' + 'TIME END OUTER: ' + str(datetime.now()) + '\n')

    return np.mean(sharpe_list_bot) - np.mean(sharpe_list_ewq)


# Optuna
#######################################################################################################

def optimize(name_test, model_name, gpu_id):
    # Auto naming
    res_timestamp = print_config()
    name_test = f"{name_test}_CPCV_{model_name}_{TIMEFRAME}_{H_TRIALS}H_{round((no_candles_for_train + no_candles_for_val) / 1000)}k"
    cwd = f"./train_results/cwd_tests/{name_test}"
    path = f"./train_results/{res_timestamp}_{name_test}/"
    if not os.path.exists(path):
        os.mkdir(path)

    with open(f"./train_results/{res_timestamp}_{name_test}/logs.txt", "w") as f:
        f.write(f"##################################  || {model_name} || ##################################")

    global study

    def obj_with_argument(trial):
        return objective(trial, name_test, model_name, cwd, res_timestamp, gpu_id)

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=SEED_CFG)
    study = optuna.create_study(
        study_name=None,
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=300,
            reduction_factor=3
        )
    )
    study.optimize(
        obj_with_argument,
        n_trials=H_TRIALS,
        catch=(ValueError,),
        callbacks=[save_best_agent]
    )

# Main
#######################################################################################

gpu_id = 0
name_model = 'ppo'
name_test = 'model'

print('\nStarting CPCV optimization with:')
print('drl algorithm:       ', name_model)
print('name_test:           ', name_test)
print('gpu_id:              ', gpu_id, '\n')

optimize(name_test, name_model, gpu_id)
