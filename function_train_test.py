"""
This python code is a function called train_and_test() which takes in multiple parameters such as trial,
price_array, tech_array, train_indices, test_indices, env, model_name, env_params, erl_params, break_step, cwd,
and gpu_id. The function first imports DRLAgent from drl_agents.elegantrl_models, BinanceProcessor from
processor_Binance, and all functions from function_finance_metrics.

The function first trains the model by creating an instance of DRLAgent_erl and passing it the environment,
price and technical arrays, and environment parameters. It then calls the get_model() method on the agent object and
passes it the model_name, gpu_id and erl_params as arguments. The function then calls the train_model() method on the
agent object and passes it the model, current working directory, and total timesteps.

The function then moves on to testing the model by creating an instance of the environment, passing it the test data
and setting the if_train parameter to False. The function then calls the DRL_prediction() method on the DRLAgent_erl
class and passes it the model_name, cwd, net_dimension, environment, and gpu_id.

Finally, the function computes the Sharpe ratios for the split by first correcting the slicing of the data,
then calling the compute_eqw() function to compute the equal-weighted Sharpe ratio, and then calling the sharpe_iid()
function to compute the Sharpe ratio for the DRL agent. The function then returns the Sharpe ratios for the DRL agent
and the equal-weighted portfolio, as well as the returns for the DRL agent.

"""

import numpy as np
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl
from processor_Binance import BinanceProcessor
from function_finance_metrics import (compute_data_points_per_year,
                                      compute_eqw,
                                      sharpe_iid)


def train_and_test(trial, price_array, tech_array, train_indices, test_indices, env, model_name, env_params, erl_params,
                   break_step, cwd, gpu_id):
    train_agent(price_array,
                tech_array,
                train_indices,
                env, model_name,
                env_params,
                erl_params,
                break_step,
                cwd,
                gpu_id)

    sharpe_bot, sharpe_eqw, drl_rets_tmp = test_agent(price_array,
                                                      tech_array,
                                                      test_indices,
                                                      env, env_params,
                                                      model_name,
                                                      cwd,
                                                      gpu_id,
                                                      erl_params,
                                                      trial)
    return sharpe_bot, sharpe_eqw, drl_rets_tmp


def train_agent(price_array, tech_array, train_indices, env, model_name, env_params, erl_params, break_step, cwd,
                gpu_id):
    print('No. Train Samples:', len(train_indices), '\n')
    price_array_train = price_array[train_indices, :]
    tech_array_train = tech_array[train_indices, :]

    agent = DRLAgent_erl(env=env,
                         price_array=price_array_train,
                         tech_array=tech_array_train,
                         env_params=env_params,
                         if_log=True)

    model = agent.get_model(model_name,
                            gpu_id,
                            model_kwargs=erl_params,
                            )

    agent.train_model(model=model,
                      cwd=cwd,
                      total_timesteps=break_step
                      )


def test_agent(price_array, tech_array, test_indices, env, env_params, model_name, cwd, gpu_id, erl_params, trial):
    print('\nNo. Test Samples:', len(test_indices))
    price_array_test = price_array[test_indices, :]
    tech_array_test = tech_array[test_indices, :]

    data_config = {
        "price_array": price_array_test,
        "tech_array": tech_array_test,
        "if_train": False,
    }

    env_instance = env(config=data_config,
                       env_params=env_params,
                       if_log=True
                       )

    net_dimension = erl_params['net_dimension']

    account_value_erl = DRLAgent_erl.DRL_prediction(
        model_name=model_name,
        cwd=cwd,
        net_dimension=net_dimension,
        environment=env_instance,
        gpu_id=gpu_id
    )
    lookback = env_params['lookback']
    indice_start = lookback - 1
    indice_end = len(price_array_test) - lookback

    data_points_per_year = compute_data_points_per_year(trial.user_attrs["timeframe"])
    account_value_eqw, eqw_rets_tmp, eqw_cumrets = compute_eqw(price_array_test, indice_start, indice_end)
    dataset_size = np.shape(eqw_rets_tmp)[0]
    factor = data_points_per_year / dataset_size
    sharpe_eqw, _ = sharpe_iid(eqw_rets_tmp, bench=0, factor=factor, log=False)

    account_value_erl = np.array(account_value_erl)
    drl_rets_tmp = account_value_erl[1:] - account_value_erl[:-1]
    sharpe_bot, _ = sharpe_iid(drl_rets_tmp, bench=0, factor=factor, log=False)

    return sharpe_bot, sharpe_eqw, drl_rets_tmp
