from torch.utils.data import Dataset
import numpy as np
from utils.tools import double_data, score_classification


class DrugDataset(Dataset):
    def __init__(self, drug_features):
        self.drug_features = drug_features

    def __len__(self):
        return self.drug_features.shape[0]

    def __getitem__(self, idx):
        drug_item = np.array(self.drug_features.iloc[idx])
        return drug_item, drug_item


class CellLineDataset(Dataset):
    def __init__(self, cell_line_features):
        self.cell_line_features = cell_line_features

    def __len__(self):
        return self.cell_line_features.shape[0]

    def __getitem__(self, idx):
        cell_line_item = np.array(self.cell_line_features.iloc[idx])
        return cell_line_item, cell_line_item


class MTLSynergyDataset(Dataset):
    def __init__(self, drugs, cell_lines, summary, syn_threshold=30, ri_threshold=50):
        self.drugs = drugs
        self.cell_lines = cell_lines
        self.summary = double_data(summary)
        self.syn_threshold = syn_threshold
        self.ri_threshold = ri_threshold

    def __len__(self):
        return self.summary.shape[0]

    def __getitem__(self, idx):
        data = self.summary.iloc[idx]
        d1_idx, d2_idx, c_idx, d1_ri, d2_ri, syn, syn_fold, sen_fold_1, sen_fold_2 = data
        d1 = np.array(self.drugs.iloc[int(d1_idx)])
        d2 = np.array(self.drugs.iloc[int(d2_idx)])
        c_exp = np.array(self.cell_lines.iloc[int(c_idx)])
        syn_label = np.array(score_classification(syn, self.syn_threshold))
        d1_label = np.array(score_classification(d1_ri, self.ri_threshold))
        return (d1, d2, c_exp, np.array(sen_fold_1)), (np.array(syn), np.array(d1_ri), syn_label, d1_label)


class OnlySynergyDataset(Dataset):
    def __init__(self, drugs, cell_lines, summary, syn_threshold=30):
        self.drugs = drugs
        self.cell_lines = cell_lines
        self.syn_threshold = syn_threshold
        self.summary = double_data(summary)

    def __len__(self):
        return self.summary.shape[0]

    def __getitem__(self, idx):
        data = self.summary.iloc[idx]
        d1_idx, d2_idx, c_idx, d1_ri, d2_ri, syn, syn_fold, sen_fold_1, sen_fold_2 = data
        d1 = np.array(self.drugs.iloc[int(d1_idx)])
        d2 = np.array(self.drugs.iloc[int(d2_idx)])
        c_exp = np.array(self.cell_lines.iloc[int(c_idx)])
        syn_label = np.array(score_classification(syn, self.syn_threshold))
        return (d1, d2, c_exp), (np.array(syn), syn_label)


class OnlySensitivityDataset(Dataset):
    def __init__(self, drugs, cell_lines, summary, ri_threshold=50):
        self.drugs = drugs
        self.cell_lines = cell_lines
        self.summary = summary
        self.ri_threshold = ri_threshold

    def __len__(self):
        return self.summary.shape[0]

    def __getitem__(self, idx):
        data = self.summary.iloc[idx]
        d1_idx, d2_idx, c_idx, d1_ri, d2_ri, syn, syn_fold, sen_fold_1, sen_fold_2 = data
        d1 = np.array(self.drugs.iloc[int(d1_idx)])
        c_exp = np.array(self.cell_lines.iloc[int(c_idx)])
        d1_label = np.array(score_classification(d1_ri, self.ri_threshold))
        return (d1, c_exp), (np.array(d1_ri), d1_label)
