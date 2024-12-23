import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size,
                          hidden_size,
                          num_layers,
                          batch_first=True,
                          dropout=0.1)
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]

        out = self.fc(out)
        out = self.sigmoid(out)

        return out
    
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size,
                        hidden_size,
                        num_layers,
                        batch_first=True,
                        bidirectional=True)

        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)

        return out
    
class CNN_GRU(nn.Module):
    def __init__(self, input_size, num_channels, hidden_size, num_layers, output_size):
        super(CNN_GRU, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=num_channels,
                               kernel_size=3, 
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels,
                               out_channels=num_channels*2,
                               kernel_size=3,
                               padding=1)
        self.pool = nn.MaxPool1d(2)
        self.gru = nn.GRU(input_size=num_channels*2,
                        hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.transpose(1, 2)

        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out
    
class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(Transformer, self).__init__()

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=0.1)
        self.fc_in = nn.Linear(input_dim, d_model)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.fc_in(src)
        src = src.permute(1, 0, 2)

        output = self.transformer.encoder(src)
        output = output.permute(1, 0, 2)
        output = self.fc_out(output[:, -1, :])

        return output