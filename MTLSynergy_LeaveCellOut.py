import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader
from torch.optim import Adam
from Dataset import MTLSynergy_LeaveCellOutDataset
from torch.nn import MSELoss
import torch
import time
import pandas as pd
from utils.tools import EarlyStopping, set_seed, CategoricalCrossEntropyLoss
from static.constant import Fold, DrugAE_SaveBase, CellAE_SaveBase
from Models import MTLSynergy, DrugAE, CellLineAE
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, cohen_kappa_score, precision_score, \
    accuracy_score

MTLSyn_LeaveCellOut_SaveBase = "save/MTLSyn_LeaveCellOut/"
MTLSyn_LeaveCellOut_Result = "result/MTLSyn_LeaveCellOut_result.txt"
device = torch.device('cuda')

HYPERPARAMETERS = { # example
    'learning_rate': 0.0001,
    'batch_size': 256,
    'hidden_neurons': [4096, 2048, 2048, 1024],
}


def fit(model, drugEncoder, cellLineEncoder, train_dataloader, train_num, optimizer, mse, cce):
    model.train()
    train_running_loss1 = 0.0
    train_running_loss2 = 0.0
    train_running_loss3 = 0.0
    train_running_loss4 = 0.0
    for i, (x, y) in enumerate(train_dataloader):
        d1_features, d2_features, c_features = x
        d1_features = d1_features.float().to(device)
        d2_features = d2_features.float().to(device)
        c_features = c_features.float().to(device)
        y1, y2, y3, y4 = y
        y1 = y1.float().to(device)
        y3 = y3.long().to(device)
        y2 = y2.float().to(device)
        y4 = y4.long().to(device)
        d1_encoder = drugEncoder(d1_features)
        d2_encoder = drugEncoder(d2_features)
        c_encoder = cellLineEncoder(c_features)
        optimizer.zero_grad()
        out1, out2, out3, out4 = model(d1_encoder, d2_encoder, c_encoder)
        loss1 = mse(out1, y1)
        loss3 = cce(out3, y3)
        loss2 = mse(out2, y2)
        loss4 = cce(out4, y4)
        num = d1_features.shape[0]
        train_running_loss1 += (loss1.item() * num)
        train_running_loss3 += (loss3.item() * num)
        train_running_loss2 += (loss2.item() * num)
        train_running_loss4 += (loss4.item() * num)
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()
    train_loss1 = train_running_loss1 / train_num
    train_loss2 = train_running_loss2 / train_num
    train_loss3 = train_running_loss3 / train_num
    train_loss4 = train_running_loss4 / train_num
    train_loss_tol = train_loss1 + train_loss2 + train_loss3 + train_loss4
    print("train loss:" + str([train_loss_tol, train_loss1, train_loss2, train_loss3, train_loss4]))
    return [train_loss_tol, train_loss1, train_loss2, train_loss3, train_loss4]


def validate(model, drugEncoder, cellLineEncoder, validation_dataloader, validation_num, mse, cce):
    model.eval()
    validation_running_loss2 = 0.0
    validation_running_loss4 = 0.0
    y_true1 = torch.Tensor().to(device)
    y_pred1 = torch.Tensor().to(device)
    y_true3 = torch.Tensor().long().to(device)
    y_pred3 = torch.Tensor().to(device)
    with torch.no_grad():
        for i, (x, y) in enumerate(validation_dataloader):
            d1_features, d2_features, c_features = x
            d1_features = d1_features.float().to(device)
            d2_features = d2_features.float().to(device)
            c_features = c_features.float().to(device)
            y1, y2, y3, y4 = y
            y1 = y1.float().to(device)
            y3 = y3.long().to(device)
            y_true1 = torch.cat((y_true1, y1), 0)
            y_true3 = torch.cat((y_true3, y3), 0)
            y2 = y2.float().to(device)
            y4 = y4.long().to(device)
            d1_encoder = drugEncoder(d1_features)
            d2_encoder = drugEncoder(d2_features)
            c_encoder = cellLineEncoder(c_features)
            out1, out2, out3, out4 = model(d1_encoder, d2_encoder, c_encoder)
            y_pred1 = torch.cat((y_pred1, out1), 0)
            y_pred3 = torch.cat((y_pred3, out3), 0)
            loss2 = mse(out2, y2)
            loss4 = cce(out4, y4)
            num = d1_features.shape[0]
            validation_running_loss2 += (loss2.item() * num)
            validation_running_loss4 += (loss4.item() * num)

        n = validation_num // 2
        y_pred1_mean = (y_pred1[0:n] + y_pred1[n:]) / 2
        y_pred3_mean = (y_pred3[0:n, ] + y_pred3[n:, ]) / 2
        validation_loss1 = mse(y_pred1_mean, y_true1[0:n]).item()
        validation_loss2 = validation_running_loss2 / validation_num
        validation_loss3 = cce(y_pred3_mean, y_true3[0:n]).item()
        validation_loss4 = validation_running_loss4 / validation_num
        validation_loss_tol = validation_loss1 + validation_loss2 + validation_loss3 + validation_loss4
        print("validation loss:" + str(
            [validation_loss_tol, validation_loss1, validation_loss2, validation_loss3, validation_loss4]))
        return [validation_loss_tol, validation_loss1, validation_loss2, validation_loss3, validation_loss4]


def test(model, drugEncoder, cellLineEncoder, test_dataloader, test_num, mse, cce):
    model.eval()
    y_true1 = torch.Tensor().to(device)
    y_pred1 = torch.Tensor().to(device)
    y_true2 = torch.Tensor().to(device)
    y_pred2 = torch.Tensor().to(device)
    y_true3 = torch.Tensor().long().to(device)
    y_pred3 = torch.Tensor().to(device)
    y_true4 = torch.Tensor().long().to(device)
    y_pred4 = torch.Tensor().to(device)
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataloader):
            d1_features, d2_features, c_features = x
            d1_features = d1_features.float().to(device)
            d2_features = d2_features.float().to(device)
            c_features = c_features.float().to(device)
            y1, y2, y3, y4 = y
            y1 = y1.float().to(device)
            y3 = y3.long().to(device)
            y_true1 = torch.cat((y_true1, y1), 0)
            y_true3 = torch.cat((y_true3, y3), 0)
            y2 = y2.float().to(device)
            y4 = y4.long().to(device)
            y_true2 = torch.cat((y_true2, y2), 0)
            y_true4 = torch.cat((y_true4, y4), 0)
            d1_encoder = drugEncoder(d1_features)
            d2_encoder = drugEncoder(d2_features)
            c_encoder = cellLineEncoder(c_features)
            out1, out2, out3, out4 = model(d1_encoder, d2_encoder, c_encoder)
            y_pred1 = torch.cat((y_pred1, out1), 0)
            y_pred3 = torch.cat((y_pred3, out3), 0)
            y_pred2 = torch.cat((y_pred2, out2), 0)
            y_pred4 = torch.cat((y_pred4, out4), 0)
        n = test_num // 2
        y_true1 = y_true1[0:n]
        y_true3 = y_true3[0:n]
        y_pred1 = (y_pred1[0:n] + y_pred1[n:]) / 2
        y_pred3 = (y_pred3[0:n, ] + y_pred3[n:, ]) / 2
        test_loss1 = mse(y_pred1, y_true1).item()
        test_loss2 = mse(y_pred2, y_true2).item()
        synergy_result = {}
        sensitivity_result = {}
        synergy_result["MSE"] = test_loss1
        sensitivity_result["MSE"] = test_loss2
        synergy_result["RMSE"] = np.sqrt(test_loss1)
        sensitivity_result["RMSE"] = np.sqrt(test_loss2)
        y_true1 = y_true1.cpu()
        y_pred1 = y_pred1.cpu()
        y_true2 = y_true2.cpu()
        y_pred2 = y_pred2.cpu()
        synergy_result["Pearsonr"] = pearsonr(y_true1, y_pred1)[0]
        sensitivity_result["Pearsonr"] = pearsonr(y_true2, y_pred2)[0]
        synergy_result["CCE"] = cce(y_pred3, y_true3).item()
        sensitivity_result["CCE"] = cce(y_pred4, y_true4).item()
        y_true3 = y_true3.cpu()
        y_pred3 = y_pred3.cpu()
        y_true4 = y_true4.cpu()
        y_pred4 = y_pred4.cpu()
        y_pred3_prob = y_pred3[:, 1]
        y_pred4_prob = y_pred4[:, 1]
        synergy_result["ROC AUC"] = roc_auc_score(y_true3, y_pred3_prob)
        sensitivity_result["ROC AUC"] = roc_auc_score(y_true4, y_pred4_prob)
        y_pred3_label = y_pred3.argmax(axis=1)
        y_pred4_label = y_pred4.argmax(axis=1)
        y_pred3_prec, y_pred3_recall, y_pred3_threshold = precision_recall_curve(y_true3, y_pred3_prob)
        y_pred4_prec, y_pred4_recall, y_pred4_threshold = precision_recall_curve(y_true4, y_pred4_prob)
        synergy_result["PR AUC"] = auc(y_pred3_recall, y_pred3_prec)
        sensitivity_result["PR AUC"] = auc(y_pred4_recall, y_pred4_prec)
        synergy_result["ACC"] = accuracy_score(y_true3, y_pred3_label)
        sensitivity_result["ACC"] = accuracy_score(y_true4, y_pred4_label)
        synergy_result["PREC"] = precision_score(y_true3, y_pred3_label)
        sensitivity_result["PREC"] = precision_score(y_true4, y_pred4_label)
        synergy_result["Kappa"] = cohen_kappa_score(y_true3, y_pred3_label)
        sensitivity_result["Kappa"] = cohen_kappa_score(y_true4, y_pred4_label)
        print("test result:\n" + "Synergy:\n" + str(synergy_result) + "\nSensitivity:\n" + str(
            sensitivity_result) + "\n")
        with open(MTLSyn_LeaveCellOut_Result, 'a') as file:
            file.write("---------- Test ----------\n")
            file.write("test result:\n" + "Synergy:\n" + str(synergy_result) + "\nSensitivity:\n" + str(
                sensitivity_result) + "\n")
        return [synergy_result, sensitivity_result]


def calculate(result, name):
    tol_result = {}
    keys = result[0].keys()
    for key in keys:
        tol_result[key] = []
    for i in range(Fold):
        for key in keys:
            tol_result[key].append(result[i][key])
    print(str(name) + " result :")
    with open(MTLSyn_LeaveCellOut_Result, 'a') as file:
        file.write(str(name) + " result :\n")
    for key in keys:
        print(str(key) + ": " + str([np.mean(tol_result[key]), np.std(tol_result[key])]))
        with open(MTLSyn_LeaveCellOut_Result, 'a') as file:
            file.write(str(key) + ": " + str([np.mean(tol_result[key]), np.std(tol_result[key])]) + "\n")


epochs = 500
patience = 100  # 50
mse = MSELoss(reduction='mean').to(device)
cce = CategoricalCrossEntropyLoss().to(device)
drugs = pd.read_csv('data/drug_features.csv')
print("drugs.shape:", drugs.shape)
cell_lines = pd.read_csv('data/cell_line_features.csv')
print("cell_lines.shape:", cell_lines.shape)
summary = pd.read_csv('data/oneil_summary_idx.csv')
print("summary.shape:", summary.shape)

drug_output = 128
cell_output = 256
drugAE = DrugAE(output_dim=drug_output).to(device)
print('---- start to load drugAE ----')
drug_path = DrugAE_SaveBase + str(drug_output) + '.pth'
drugAE.load_state_dict(torch.load(drug_path))
drugAE.eval()
cell_path = CellAE_SaveBase + str(cell_output) + '.pth'
cellLineAE = CellLineAE(output_dim=cell_output).to(device)
print('---- start to load cellLineAE ----')
cellLineAE.load_state_dict(torch.load(cell_path))
cellLineAE.eval()
print("--------- MTL-Model with drugAE(" + str(drug_output) + ") cellLineAE(" + str(cell_output) + ") ---------")
with open(MTLSyn_LeaveCellOut_Result, 'a') as file:
    file.write("--------- MTL-Model with drugAE(" + str(drug_output) + ") cellLineAE(" + str(
        cell_output) + ") ---------\n")
result_per_fold = []
model_input = drug_output + cell_output

for fold_test in range(0, Fold):
    print("----- Test Fold " + str(fold_test) + " -----")
    with open(MTLSyn_LeaveCellOut_Result, 'a') as file:
        file.write("----- Test Fold " + str(fold_test) + " -----\n")
        file.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + "\n")
    set_seed(seed=1)
    test_summary = summary.loc[summary['fold'] == fold_test]
    train_val_summary = summary.loc[summary['fold']!=fold_test]
    validation_summary = train_val_summary.sample(frac=0.1, replace=False, random_state=1, axis=0)
    train_summary = train_val_summary.loc[~train_val_summary.index.isin(validation_summary.index)]
    train_dataset = MTLSynergy_LeaveCellOutDataset(drugs, cell_lines, train_summary)
    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=True,
                              num_workers=2, pin_memory=True)
    validation_dataset = MTLSynergy_LeaveCellOutDataset(drugs, cell_lines, validation_summary)
    validation_num = len(validation_dataset)
    validation_loader = DataLoader(validation_dataset, batch_size=HYPERPARAMETERS['batch_size'],
                                   shuffle=False)  # no shuffle
    test_dataset = MTLSynergy_LeaveCellOutDataset(drugs, cell_lines, test_summary)
    test_num = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=HYPERPARAMETERS['batch_size'],
                             shuffle=False)  # no shuffle

    model = MTLSynergy(HYPERPARAMETERS['hidden_neurons'], model_input).to(device)
    model_es = EarlyStopping(patience=patience)
    optimizer = Adam(model.parameters(), lr=HYPERPARAMETERS['learning_rate'])
    train_loss = []
    validation_loss = []
    start_time = time.time()
    save_path = MTLSyn_LeaveCellOut_SaveBase + "fold_" + str(fold_test) + ".pth"
    for epoch in range(epochs):
        train_result = fit(model, drugAE.encoder, cellLineAE.encoder, train_loader, train_num, optimizer, mse, cce)
        train_loss.append(train_result)
        validation_result = validate(model, drugAE.encoder, cellLineAE.encoder, validation_loader, validation_num,
                                     mse, cce)
        validation_loss.append(validation_result)

        model_es(validation_result[0], model, save_path)
        if model_es.early_stop:
            with open(MTLSyn_LeaveCellOut_Result, 'a') as file:
                file.write("When in epoch " + str(epoch - patience + 1) + ":\n")
                file.write("Validation loss:" + str(validation_loss[epoch - patience]) + "\n")
                file.write("Best loss:" + str(model_es.best_loss) + "\n")
            break

    model = MTLSynergy(HYPERPARAMETERS['hidden_neurons'], model_input).to(device)
    model.load_state_dict(torch.load(save_path))
    test_result = test(model, drugAE.encoder, cellLineAE.encoder, test_loader, test_num, mse, cce)
    result_per_fold.append(test_result)
result = np.array(result_per_fold)
with open(MTLSyn_LeaveCellOut_Result, 'a') as file:
    file.write("------------------ Calculate ---------------\n")
calculate(result[:, 0], "Synergy")
calculate(result[:, 1], "Sensitivity")
with open(MTLSyn_LeaveCellOut_Result, 'a') as file:
    file.write("\n")
