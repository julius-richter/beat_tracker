import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class ModelBoeck(nn.Module):
    def __init__(self):
        super(ModelBoeck, self).__init__()
        
        # Model parameters
        self.input_size = 120
        self.output_size = 2
        self.num_layers = 3
        self.hidden_size = 25     
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        
        # Recurrent layer 
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, 
                            bidirectional=self.bidirectional,
                            batch_first=True)
        
        # Read out layer
        self.fc = nn.Linear(self.num_directions * self.hidden_size, self.output_size)       
        
    def forward(self, x): 
                
        lstm_out, _ = self.lstm(x) 
                
        fc_out = self.fc(lstm_out)
        
        y = F.log_softmax(fc_out, dim=2)
        
        return torch.transpose(y, 1, 2)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, nhid, levels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.num_channels = [nhid] * levels
        self.tcn = TemporalConvNet(input_size, self.num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(self.num_channels[-1], output_size)

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output).double()        
        output = F.log_softmax(output, dim=2).transpose(1, 2)
        return output