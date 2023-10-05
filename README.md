# MTLSynergy

## Requirements

- python=3.7
- cuda=10.2
- pytorch=1.8.1
- sklearn=1.0.2
- pandas=1.3.5

## Start

Run the AEtrain.py first to pre-train a drug encoder and a cell line encoder, and then run the MTLSynergytrain.py to train the model.

## Data

**drugs.csv**:  Information of 3118 drugs.

**cell_lines.csv**:  Information of 175 cell lines.

**drug_features.csv**:  Features of  3118 drugs, 1213-dimensional vector for each drug.

**cell_line_features.csv**:  Features of 175 cell lines, 5000-dimensional vector for each cell lines.

**oneil_summary_idx.csv**:  22 737 samples from O'Neil，each sample consists of two drugs id, a cell line id, synergy score of the drug combination on the cell line, respective sensitivity scores of the two drugs on the cell line.  

## Training files

**AEtrain.py**: used to pre-train a drug encoder and a cell line encoder.

**MTLSynergytrain.py**: used to train MTLSynergy in the *Leave Drug Combinations Out* scenario.

**MTLSynergy_LeaveCellOut.py**: used to train MTLSynergy in the *Leave Cell Lines Out* scenario.

**MTLSynergy_LeaveDrugOut.py**: used to train MTLSynergy in the *Leave Drugs Out* scenario.

**GBMtrain.py**: used to train Gradient Boosting Machine in the *Leave Drug Combinations Out* scenario.

**GBM_LeaveCellOut.py**: used to train Gradient Boosting Machine in the *Leave Cell Lines Out* scenario.

**GBM_LeaveDrugOut.py**: used to train Gradient Boosting Machine in the *Leave Drugs Out* scenario.

**RFtrain.py**: used to train Random Forest in the *Leave Drug Combinations Out* scenario.

**RF_LeaveCellOut.py**: used to train Random Forest in the *Leave Cell Lines Out* scenario.

**RF_LeaveDrugOut.py**: used to train Random Forest in the *Leave Drugs Out* scenario.



## Source code of the comparative methods

PRODeepSyn: https://github.com/TOJSSE-iData/PRODeepSyn

TranSynergy: https://github.com/qiaoliuhub/drug_combination

AuDnnSynergy: The authors did not provide the source code.

DeepSynergy: https://github.com/KristinaPreuer/DeepSynergy