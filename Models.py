import torch
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d, Dropout, Softmax
from static.constant import DrugAE_InputDim, DrugAE_OutputDim, CELLAE_InputDim, CellAE_OutputDim, MTLSynergy_InputDim
from utils.tools import init_weights


class DrugAE(Module):
    def __init__(self, input_dim=DrugAE_InputDim, output_dim=DrugAE_OutputDim):
        super(DrugAE, self).__init__()
        if output_dim == 32 or output_dim == 64:
            hidden_dim = 256
        elif output_dim == 128 or output_dim == 256:
            hidden_dim = 512
        else:
            hidden_dim = 1024
        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_dim, output_dim),
        )
        self.decoder = Sequential(
            Linear(output_dim, hidden_dim),
            ReLU(True),
            Linear(hidden_dim, input_dim),
        )
        init_weights(self._modules)

    def forward(self, input):
        x = self.encoder(input)
        y = self.decoder(x)
        return y


class CellLineAE(Module):
    def __init__(self, input_dim=CELLAE_InputDim, output_dim=CellAE_OutputDim):
        super(CellLineAE, self).__init__()
        if output_dim == 128 or output_dim == 256:
            hidden_dim = 512
        elif output_dim == 512:
            hidden_dim = 1024
        else:
            hidden_dim = 4096
        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_dim, output_dim),
        )
        self.decoder = Sequential(
            Linear(output_dim, hidden_dim),
            ReLU(True),
            Linear(hidden_dim, input_dim)
        )
        init_weights(self._modules)

    def forward(self, input):
        x = self.encoder(input)
        y = self.decoder(x)
        return y


class MTLSynergy(Module):
    def __init__(self, hidden_neurons, input_dim=MTLSynergy_InputDim):
        super(MTLSynergy, self).__init__() 
        self.drug_cell_line_layer = Sequential(
            Linear(input_dim, hidden_neurons[0]),
            BatchNorm1d(hidden_neurons[0]),
            ReLU(True),
            Linear(hidden_neurons[0], hidden_neurons[1]),
            ReLU(True)
        )
        self.synergy_layer = Sequential(
            Linear(2 * hidden_neurons[1], hidden_neurons[2]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[2], 128),
            ReLU(True)
        )
        self.sensitivity_layer = Sequential(
            Linear(hidden_neurons[1], hidden_neurons[3]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[3], 64),
            ReLU(True)
        )
        self.synergy_out_1 = Linear(128, 1)
        self.synergy_out_2 = Sequential(Linear(128, 2), Softmax(dim=1))
        self.sensitivity_out_1 = Linear(64, 1)
        self.sensitivity_out_2 = Sequential(Linear(64, 2), Softmax(dim=1))
        init_weights(self._modules)

    def forward(self, d1, d2, c_exp):
        d1_c = self.drug_cell_line_layer(torch.cat((d1, c_exp), 1))
        d2_c = self.drug_cell_line_layer(torch.cat((d2, c_exp), 1))
        d1_sen = self.sensitivity_layer(d1_c)
        syn = self.synergy_layer(torch.cat((d1_c, d2_c), 1))
        syn_out_1 = self.synergy_out_1(syn)
        syn_out_2 = self.synergy_out_2(syn)
        d1_sen_out_1 = self.sensitivity_out_1(d1_sen)
        d1_sen_out_2 = self.sensitivity_out_2(d1_sen)
        return syn_out_1.squeeze(-1), d1_sen_out_1.squeeze(-1), syn_out_2, d1_sen_out_2

