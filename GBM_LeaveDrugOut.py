import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
import time
import random
from utils.tools import double_data, calculate

drugs = pd.read_csv('data/drug_features.csv')
print("drugs.shape:", drugs.shape)
cell_lines = pd.read_csv('data/cell_line_features.csv')
print("cell_lines.shape:", cell_lines.shape)
summary = pd.read_csv('data/oneil_summary_idx.csv')
print("summary.shape:", summary.shape)
FILE_URL = "result/GBM_LeaveDrugOut_result.txt"


class DataLoader:
    def __init__(self, drugs, cell_lines, summary, test_fold, syn_threshold=30):
        self.drugs = drugs
        self.cell_lines = cell_lines
        self.summary = double_data(summary)
        self.syn_threshold = syn_threshold
        self.summary_test = self.summary.loc[(self.summary['sen_fold_1'] == test_fold)|(self.summary['sen_fold_2']==test_fold)]
        self.summary_train = self.summary.loc[~self.summary.index.isin(self.summary_test.index)]
        self.length_train = self.summary_train.shape[0]
        print("train:", self.length_train)
        self.length_test = self.summary_test.shape[0]
        print("test:", self.length_test)

    def syn_map(self, x):
        return 1 if x > self.syn_threshold else 0

    def get_samples(self, flag, method):
        if flag == 0:  # train data
            summary = self.summary_train
        else:  # test data
            summary = self.summary_test
        d1_idx = summary.iloc[:, 0]
        d2_idx = summary.iloc[:, 1]
        c_idx = summary.iloc[:, 2]
        d1 = np.array(self.drugs.iloc[d1_idx])
        d2 = np.array(self.drugs.iloc[d2_idx])
        c_exp = np.array(self.cell_lines.iloc[c_idx])
        X = np.concatenate((d1, d2, c_exp), axis=1)
        if method == 0:  # regression
            y = np.array(summary.iloc[:, 5])
        else:  # classification
            y = np.array(summary.iloc[:, 5].apply(lambda s: self.syn_map(s)))
        return X, y


Fold = 5

print("----------- Regression ----------")
with open(FILE_URL, 'a') as file:
    file.write("---------------------- Regression ---------------------\n")
result_r = []
for fold_test in range(0, Fold):
    print("---------- Test Fold " + str(fold_test) + " ----------")
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    random.seed(1)
    np.random.seed(1)
    with open(FILE_URL, 'a') as file:
        file.write("---------- Test Fold " + str(fold_test) + " ----------\n")
        file.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + "\n")
    sampelData = DataLoader(drugs, cell_lines, summary, test_fold=fold_test)
    x_train, y_train = sampelData.get_samples(0, 0)
    x_test, syn_true_value = sampelData.get_samples(1, 0)
    hyper_params = {'n_estimators': [512, 1024], 'learning_rate': [0.1, 0.01]}
    gbr = GradientBoostingRegressor(subsample=0.8, max_depth=25, min_samples_split=100, min_samples_leaf=20,
                                    max_features='sqrt', random_state=1)
    grid_cv = GridSearchCV(gbr, param_grid=hyper_params, scoring='neg_mean_squared_error', verbose=10, cv=4)

    grid_cv.fit(x_train, y_train)
    syn_pred_value = grid_cv.predict(x_test)
    n = sampelData.length_test // 2
    syn_true_value = syn_true_value[0:n]
    syn_pred_value = (syn_pred_value[0:n] + syn_pred_value[n:]) / 2
    syn_metrics = {}
    syn_metrics['MSE'] = mean_squared_error(syn_true_value, syn_pred_value)
    syn_metrics['RMSE'] = np.sqrt(syn_metrics['MSE'])
    syn_metrics["Pearsonr"] = pearsonr(syn_true_value, syn_pred_value)[0]
    result_r.append(syn_metrics)
    print("syn_metrics:", syn_metrics)
    with open(FILE_URL, 'a') as file:
        file.write("syn_metrics:" + str(syn_metrics) + "\n")
calculate(np.array(result_r), "regression", Fold, FILE_URL)