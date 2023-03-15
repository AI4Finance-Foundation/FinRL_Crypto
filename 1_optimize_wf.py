"""This script is used for training and evaluating a reinforcement learning agent for trading on the Alpaca platform
using Optuna for hyperparameter optimization and joblib for parallel execution of trials.

The script imports various modules including joblib, optuna, datetime, pickle, sys, distutils.dir_util,
environment_Alpaca, function_CPCV, function_train_test, config_main.

The script also contains a class 'bcolors' which is used to color the output text in the terminal.

The script defines a function 'print_config' which prints the configuration of the current trial including the time
frame, number of samples, number of trials, and number of splits. It also returns a timestamp used for naming the
results folder.

The function 'set_Pandas_Timedelta' is used to set the timedelta for the Pandas dataframe based on the selected time
frame.

The function 'save_best_agent' is used to save the best agent obtained from the trials. It copies the agent from the
working directory and saves it in the results folder. It also pickles the trial information to avoid errors where
params are not copied.

The function 'sample_hyperparams' is used for sampling the hyperparameters for the trials. It returns a dictionary of
the hyperparameters.

The main function is defined outside of this script, and it sets up the environment, runs the trials, and saves the
results."""

import joblib
import optuna
import datetime
import pickle
import os

from distutils.dir_util import copy_tree

import pandas as pd

from environment_Alpaca import CryptoEnvAlpaca
from function_train_test import train_and_test
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
    print('\n' + bcolors.HEADER + '##### Launched hyperparameter optimization with K-Cross Validation   #####' + bcolors.ENDC + '\n')
    print('TIMEFRAME                  ', TIMEFRAME)
    print('TRAIN SAMPLES              ', no_candles_for_train)
    print('TRIALS NO.                 ', H_TRIALS)
    print('N                          ', N_GROUPS)
    print('K slash ', K_TEST_GROUPS)
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


def set_pickle_attributes(trial, model_name, TIMEFRAME, TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE,
                          TICKER_LIST, TECHNICAL_INDICATORS_LIST, name_folder, name_test, study):
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


def write_logs(name_folder, model_name, trial, cwd, erl_params, env_params):
    path_logs = './train_results/' + name_folder + '/logs.txt'
    with open(path_logs, 'a') as f:
        f.write('\n' + 'MODEL NAME: ' + model_name + '\n')
        f.write('TRIAL NUMBER: ' + str(trial.number) + '\n')
        f.write('CWD: ' + cwd + '\n')
        f.write(str(erl_params) + '\n')
        f.write(str(env_params) + '\n')
        f.write('\n' + 'TIME START OUTER: ' + str(datetime.now()) + '\n')
    return path_logs


def objective(trial, name_test, model_name, cwd, res_timestamp, gpu_id):
    # Set full name_folder
    name_folder = res_timestamp + '_' + name_test

    set_pickle_attributes(trial, model_name, TIMEFRAME, TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE,
                          TICKER_LIST, TECHNICAL_INDICATORS_LIST, name_folder, name_test, study)

    # Sample set of hyperparameters
    erl_params, env_params = sample_hyperparams(trial)

    # Load data from hard disk
    data_from_processor, price_array, tech_array, time_array = load_saved_data(TIMEFRAME, no_candles_for_train)

    # initiate logs for tracking behaviour during training
    path_logs = write_logs(name_folder, model_name, trial, cwd, erl_params, env_params)

    # WF Split function eval (single walk-forward set so no averaging compared to KCV and CPCV
    #######################################################################################################
    #######################################################################################################

    env = CryptoEnvAlpaca
    break_step = erl_params['break_step']

    sharpe_list_bot = []
    sharpe_list_ewq = []
    drl_rets_val_list = []

    # select indices for train
    train_indices = list(range(1, no_candles_for_train))
    test_indices = list(range(no_candles_for_train, no_candles_for_train + no_candles_for_val - 1))

    # Train and test
    sharpe_bot, sharpe_eqw, drl_rets_tmp = train_and_test(trial, price_array, tech_array, train_indices,
                                                          test_indices, env, model_name, env_params,
                                                          erl_params, break_step, cwd, gpu_id)

    with open(path_logs, 'a') as f:
        f.write('BOT:         ' + str(sharpe_bot) + '\n')
        f.write('HODL:        ' + str(sharpe_eqw) + '\n')
        f.write('TIME END INNER: ' + str(datetime.now()) + '\n\n')

    # Fill the backtesting prediction matrix
    drl_rets_val_list.append(drl_rets_tmp)
    trial.set_user_attr("price_array", price_array)
    trial.set_user_attr("tech_array", tech_array)
    trial.set_user_attr("time_array", time_array)

    # Matrices
    trial.set_user_attr("drl_rets_val_list", drl_rets_val_list)

    # Interesting values
    trial.set_user_attr("sharpe_list_bot", sharpe_bot)
    trial.set_user_attr("sharpe_list_ewq", sharpe_eqw)

    with open(path_logs, 'a') as f:
        f.write('\nHYPERPARAMETER EVAL || SHARPE BOT    :  ' + str(sharpe_bot) + '\n')
        f.write('HYPERPARAMETER EVAL || SHARPE HODL     : ' + str(sharpe_eqw) + '\n')
        f.write('DIFFERENCE                                 : ' + str(sharpe_bot - sharpe_eqw) + '\n')
        f.write('\n' + 'TIME END OUTER: ' + str(datetime.now()) + '\n')

    return sharpe_bot - sharpe_eqw


# Optuna
#######################################################################################################


def optimize(name_test, model_name, gpu_id):
    # Auto naming
    res_timestamp = print_config()
    name_test = f"{name_test}_WF_{model_name}_{TIMEFRAME}_{H_TRIALS}H_{round((no_candles_for_train + no_candles_for_val) / 1000)}k"
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

print('\nStarting WF optimization with:')
print('drl algorithm:       ', name_model)
print('name_test:           ', name_test)
print('gpu_id:              ', gpu_id, '\n')

optimize(name_test, name_model, gpu_id)
