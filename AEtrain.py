import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.data import DataLoader
from torch.optim import Adam
from Dataset import DrugDataset, CellLineDataset
from Models import DrugAE, CellLineAE
from torch.nn import MSELoss
import torch
import time
import pandas as pd
from utils.tools import EarlyStopping, set_seed
from static.constant import DrugAE_OutputDim_Optional, CellAE_OutputDim_Optional, DrugAE_SaveBase, CellAE_SaveBase, \
    DrugAE_Result, CellLineAE_Result

device = torch.device('cuda')


def fit(model, train_dataloader, train_num, optimizer, criterion):
    print('---Training---')
    model.train()
    train_running_loss = 0.0
    for i, (x, y) in enumerate(train_dataloader):
        data, target = x.float().to(device), y.float().to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += (loss.item() * x.shape[0])
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / train_num
    return train_loss


def validate(model, validation_dataloader, validation_num, criterion):
    model.eval()
    validation_running_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(validation_dataloader):
            data, target = x.float().to(device), y.float().to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            validation_running_loss += (loss.item() * x.shape[0])
        validation_loss = validation_running_loss / validation_num
        return validation_loss


# drug AE train
drug_features_data = pd.read_csv('data/drug_features.csv')
drug_num = drug_features_data.shape[0]
print("drugs num:", drug_num)
drug_bz = 32
drug_lr = 0.0001
drug_epochs = 3000
drug_patience = 100
for drug_outputdim in DrugAE_OutputDim_Optional:
    set_seed(1)
    drugAE = DrugAE(output_dim=drug_outputdim).to(device)
    drug_optimizer = Adam(drugAE.parameters(), lr=drug_lr)
    drug_loss_fn = MSELoss(reduction='mean').to(device)
    drug_dataset = DrugDataset(drug_features_data)
    drug_feature_loader = DataLoader(drug_dataset, batch_size=drug_bz, shuffle=True)
    validation_loader = DataLoader(drug_dataset, batch_size=drug_bz)
    drug_es = EarlyStopping(patience=drug_patience)
    path = DrugAE_SaveBase + str(drug_outputdim) + ".pth"
    with open(DrugAE_Result, 'a') as file:
        file.write("---- start drugAE_" + str(drug_outputdim) + " train ----\n")
        file.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + "\n")
    if os.path.exists(path):
        print('---- start to load drugAE_' + str(drug_outputdim) + ' ----')
        drugAE.load_state_dict(torch.load(path))
        drug_es.best_loss = validate(drugAE, validation_loader, drug_num, drug_loss_fn)
        with open(DrugAE_Result, 'a') as file:
            file.write("---- Before loss:" + str(drug_es.best_loss) + "\n")
    # drug_train_loss = []
    print("---- start drugAE_" + str(drug_outputdim) + " train ----")
    for epoch in range(drug_epochs):
        print(f"Epoch {epoch + 1} of {drug_epochs}")
        if drug_es.early_stop:
            break
        train_epoch_loss = fit(
            drugAE, drug_feature_loader, drug_num, drug_optimizer, drug_loss_fn
        )
        # drug_train_loss.append(train_epoch_loss)
        validation_loss = validate(drugAE, validation_loader, drug_num, drug_loss_fn)
        drug_es(validation_loss, drugAE, path)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Best Loss:{drug_es.best_loss:.4f}")
    with open(DrugAE_Result, 'a') as file:
        file.write("Best Loss:" + str(drug_es.best_loss) + "\n")

# cell line AE train
cell_line_features_data = pd.read_csv('data/cell_line_features.csv')
cell_line_num = cell_line_features_data.shape[0]
print("cell lines num:", cell_line_num)
cell_line_bz = 32
cell_line_lr = 0.0001
cell_line_epochs = 1500
cell_line_patience = 100

for cell_outputdim in CellAE_OutputDim_Optional:
    set_seed(1)
    cellLineAE = CellLineAE(output_dim=cell_outputdim).to(device)
    cell_line_optimizer = Adam(cellLineAE.parameters(), lr=cell_line_lr)
    cell_line_loss_fn = MSELoss(reduction='mean').to(device)
    cell_line_dataset = CellLineDataset(cell_line_features_data)
    cell_line_feature_loader = DataLoader(cell_line_dataset, batch_size=cell_line_bz, shuffle=True)
    validation_loader = DataLoader(cell_line_dataset, batch_size=cell_line_bz)
    cell_line_es = EarlyStopping(patience=cell_line_patience)
    path = CellAE_SaveBase + str(cell_outputdim) + ".pth"
    with open(CellLineAE_Result, 'a') as file:
        file.write("---- start cellLineAE_" + str(cell_outputdim) + ' train ----\n')
        file.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + "\n")
    if os.path.exists(path):
        print('---- start to load cellLineAE_' + str(cell_outputdim) + ' ----')
        cellLineAE.load_state_dict(torch.load(path))
        cell_line_es.best_loss = validate(cellLineAE, validation_loader, cell_line_num, cell_line_loss_fn)
        with open(CellLineAE_Result, 'a') as file:
            file.write("---- Before loss:" + str(cell_line_es.best_loss) + "\n")
    # cell_line_train_loss = []
    print("---- start cellLineAE_" + str(cell_outputdim) + ' train ----')
    for epoch in range(cell_line_epochs):
        print(f"Epoch {epoch + 1} of {cell_line_epochs}")
        if cell_line_es.early_stop:
            break
        train_epoch_loss = fit(
            cellLineAE, cell_line_feature_loader, cell_line_num, cell_line_optimizer, cell_line_loss_fn
        )
        # cell_line_train_loss.append(train_epoch_loss)
        validation_loss = validate(cellLineAE, validation_loader, cell_line_num, cell_line_loss_fn)
        cell_line_es(validation_loss, cellLineAE, path)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Best Loss:{cell_line_es.best_loss:.4f}")
    with open(CellLineAE_Result, 'a') as file:
        file.write("Best Loss:" + str(cell_line_es.best_loss) + "\n")