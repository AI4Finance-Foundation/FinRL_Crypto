"""This code defines a class named 'StudyAnalyzer' that takes in a file path to a pickle file containing the results
of an Optuna optimization study. The class has one method 'analyze' which loads the study from the pickle file,
retrieves some lists of data from the best trial, plots some Optuna visualization, prints some information and calls
some external functions."""

import optuna
import joblib
from function_finance_metrics import *
import os
import scipy.stats as stats


class StudyAnalyzer:
    def __init__(self, pickle_result: str):
        self.pickle_result = pickle_result
        self.study = joblib.load(f'train_results/{pickle_result}/study.pkl')
        self.best_trial = self.study.best_trial

        self.image_path = 'plots_and_metrics'
        os.makedirs(self.image_path, exist_ok=True)

    def analyze(self):
        # get lists
        self.sharpe_list_drl = self.best_trial.user_attrs['sharpe_list_bot']
        self.sharpe_list_hodl = self.best_trial.user_attrs['sharpe_list_ewq']

        # Plot Optuna optimization
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.show()
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        fig.show()
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.show()

        print(self.sharpe_list_drl)

        print('\nSharpe DRL:')
        print(np.round(self.sharpe_list_drl, 2))

        print('\nSharpe HODL:')
        print(np.round(self.sharpe_list_hodl, 2), '\n')

        plot_pdf(self.sharpe_list_drl, self.sharpe_list_hodl, self.image_path + "/pdf_validation_sharpe",
                 if_range_hodl=True)
        print('###### 95% confidence interval ######')
        print(np.round(mean_confidence_interval(self.sharpe_list_drl), 2))
        print(mean_confidence_interval(self.sharpe_list_hodl))
        print(np.var(self.sharpe_list_drl), np.var(self.sharpe_list_hodl))
        print(stats.ttest_ind(a=self.sharpe_list_drl, b=self.sharpe_list_hodl, equal_var=True))


if __name__ == "__main__":
    pickle_result = input("Enter pickle result dir: ")
    study_analyzer = StudyAnalyzer(pickle_result)
    study_analyzer.analyze()