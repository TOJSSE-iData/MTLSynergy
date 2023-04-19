import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import pandas as pd


class EarlyStopping():
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, save_path):
        if self.best_loss == None:
            self.best_loss = val_loss
            torch.save(model.state_dict(), save_path)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            # save weights
            torch.save(model.state_dict(), save_path)
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def set_seed(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # set random seed for CPU
    torch.cuda.manual_seed(seed)  # set random seed for GPU


def double_data(data):
    double_summary = pd.DataFrame()
    double_summary['drug_row_idx'] = data['drug_col_idx']
    double_summary['drug_col_idx'] = data['drug_row_idx']
    double_summary['cell_line_idx'] = data['cell_line_idx']
    double_summary['ri_row'] = data['ri_col']
    double_summary['ri_col'] = data['ri_row']
    double_summary['synergy_loewe'] = data['synergy_loewe']
    double_summary['syn_fold'] = data['syn_fold']
    double_summary['sen_fold_1'] = data['sen_fold_2']
    double_summary['sen_fold_2'] = data['sen_fold_1']
    result = pd.concat([data, double_summary], axis=0)
    return result


def init_weights(modules):
    for n, m in modules.items():
        if isinstance(m, torch.nn.Sequential):
            for layer in m:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight, 1.0)
                    torch.nn.init.constant_(layer.bias, 0.0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0.0)


def filter1(data, fold_test, flag=0):
    x, y = data
    d1_features, d2_features, c_features, sen_fold = x
    y1, y2, y3, y4 = y
    if flag == 0:
        remain = np.where(sen_fold != fold_test)
    else:
        remain = np.where(sen_fold == fold_test)
    return (d1_features[remain], d2_features[remain], c_features[remain]), (
        y1[remain], y2[remain], y3[remain], y4[remain])


def filter2(data, fold_test, flag=0):
    x, y = data
    d1_features, d2_features, c_features, sen_fold = x
    y1, y2 = y
    if flag == 0:
        remain = np.where(sen_fold != fold_test)
    else:
        remain = np.where(sen_fold == fold_test)
    return (d1_features[remain], d2_features[remain], c_features[remain]), (y1[remain], y2[remain])


def score_classification(x, threshold):
    return 1 if x > threshold else 0


def calculate(result, name, fold_num, save_path):
    tol_result = {}
    keys = result[0].keys()
    for key in keys:
        tol_result[key] = []
    for i in range(fold_num):
        for key in keys:
            tol_result[key].append(result[i][key])
    print(str(name) + " result :")
    with open(save_path, 'a') as file:
        file.write(str(name) + " result :\n")
    for key in keys:
        print(str(key) + ": " + str([np.mean(tol_result[key]), np.std(tol_result[key])]))
        with open(save_path, 'a') as file:
            file.write(str(key) + ": " + str([np.mean(tol_result[key]), np.std(tol_result[key])]) + "\n")



class CategoricalCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true, y_pred.shape[1])
        y_pred = torch.clamp(y_pred, 1e-9, 1.0)
        tol_loss = -torch.sum(y_true * torch.log(y_pred), dim=1)
        loss = torch.mean(tol_loss, dim=0)
        return loss