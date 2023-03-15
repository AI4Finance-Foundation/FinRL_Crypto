import joblib
import seaborn as sns
import math
from function_finance_metrics import *
from function_PBO import pbo
from config_main import *


# Functions
#######################################################################################################

def add_samples_equify_array_length(return_arrays):
    no_arrays = np.shape(return_arrays)[0]
    max_length = -math.inf
    for row in return_arrays:
        length = np.shape(row)[0]
        if length > max_length:
            max_length = length
    new_ret_array = np.empty([max_length, no_arrays])
    for idx, row in enumerate(return_arrays):
        while np.shape(row)[0] != max_length:
            row = np.append(row, row[-1])
        new_ret_array[:, idx] = row
    return new_ret_array


def main_metric_pbo_analysis(x):
    sharpe, _ = sharpe_iid(x, bench=0, factor=1, log=False)
    return sharpe


def load_validated_model(pickle_result):
    study = joblib.load(f'train_results/{pickle_result}/study.pkl')
    best_trial_number = study.best_trial.number
    print('BEST TRIAL: ', best_trial_number)

    trials = study.trials
    number_of_trials = len(trials) - 1
    name_test = trials[0].user_attrs['name_test']
    timeframe = trials[0].user_attrs['timeframe']
    model_name = trials[0].user_attrs['model_name']
    to_beat_sharpe = np.mean(trials[0].user_attrs['sharpe_list_ewq'])
    return best_trial_number, study, trials, model_name, number_of_trials, name_test, timeframe, to_beat_sharpe


def build_matrix_M_splits(trials, number_of_trials):
    matrix_cumrets_val = []
    for i in range(number_of_trials):
        trial = trials[i]
        drl_rets_val_list = trial.user_attrs['drl_rets_val_list']
        drl_rets_val_list= add_samples_equify_array_length(drl_rets_val_list)
        rets_single_trial = np.vstack(drl_rets_val_list)
        rets_single_trial = np.mean(rets_single_trial, axis=0)
        matrix_cumrets_val.append(rets_single_trial)
    matrix_cumrets_val = np.transpose(np.vstack(matrix_cumrets_val))
    return matrix_cumrets_val


def build_matrix_M_no_splits(trials, number_of_trials):
    matrix_cumrets_val = []
    for i in range(number_of_trials):
        drl_rets_val_list_single = np.array(trials[i].user_attrs['drl_rets_val_list'])
        drl_rets = drl_rets_val_list_single[:-1] / drl_rets_val_list_single[1:] - 1
        drl_rets = np.mean(drl_rets, axis=0)
        matrix_cumrets_val.append(drl_rets)
    matrix_cumrets_val = np.transpose(np.vstack(matrix_cumrets_val))
    return matrix_cumrets_val


# Inputs: Results and number of splits of matrix M
#######################################################################################################

pickle_results = [
                  "res_2023-01-23__17_07_49_model_KCV_ppo_5m_3H_20005k",
                  "res_2023-01-23__16_44_30_model_CPCV_ppo_5m_3H_20k"
                  ]
S = 14

# Execution
#######################################################################################################
#######################################################################################################
#######################################################################################################

model_names = []
pbo_results = []
M_matrices = []
logits_list = []
for count, result in enumerate(pickle_results):
    print('Result No.: ', count)
    print(result)
    best_trial_number, study, trials, model_name, number_of_trials, name_test, timeframe, to_beat_sharpe = load_validated_model(
        result)

    # if count == 0:
    #     M = build_matrix_M_no_splits(trials, number_of_trials)
    # else:
    M = build_matrix_M_splits(trials, number_of_trials)

    pbox = pbo(M,
               S=S,
               metric_func=main_metric_pbo_analysis,
               name_exp=name_test,
               threshold=to_beat_sharpe, n_jobs=4,
               plot=False, verbose=False, hist=False)
    print('EWQ Sharpe to Beat: ', to_beat_sharpe)

    logits = pbox.logits

    print('Min. logit:  ', min(logits))
    print('Max. logit:  ', max(logits))
    print('Mean logits: ', np.mean(logits))

    logits_list.append(logits)
    phi_self = np.array([1.0 if lam <= 0 else 0.0 for lam in logits]) / len(logits)
    pbo_self = np.sum(phi_self)

    print('PBO: ', pbo_self * 100, '%\n')

    pbo_results.append(pbo_self * 100)
    model_names.append(model_name)
    M_matrices.append(M)

# Plot PBO
#######################################################################################################
#######################################################################################################
#######################################################################################################

#model_names_list = ['ppo', 'sac', 'td3']
model_names = ['WF', 'KCV', 'CPCV']

model_names = [name.upper() for name in model_names]
sns.set(rc={'figure.figsize': (10, 6)})
sns.set(font_scale=2)
sns.set_style('whitegrid')
for i in range(len(pbo_results)):
    ax = sns.distplot(
        logits_list[i],
        label=model_names[i],
        kde_kws=dict(linewidth=3),
        hist=False
        )
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(3)


# plot text
axes = plt.gca()
y_min, y_max = axes.get_ylim()
x_min, x_max = axes.get_xlim()
print('Lower/Upper bound axis', y_min, y_max)
for i in range(len(pbo_results)):
    pbo_i = pbo_results[i]
    figure_string = "{}\n$p$={}%".format(model_names[i], format(pbo_i, '.1f'))
    plt.text(x_min + 0.2, y_max / 1.2 - (i * (y_max / 1 / len(model_names))), figure_string)

# Final stuff
plt.axvline(0, c="r", ls="--", linewidth=3)
plt.legend(frameon=False, ncol=len(model_names), loc='upper right', bbox_to_anchor=(1, 1.17), fontsize=22)
ax.set(xlabel="Logits")
ax.set(ylabel="Distribution (%)")
plt.savefig("./plots_and_metrics/multiple_logits_dist", bbox_inches='tight')
