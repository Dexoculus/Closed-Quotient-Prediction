import torch
import torch.nn as nn
import torch.nn.functional as F

class model_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(model_GRU, self).__init__()
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
    
class model_BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(model_BiGRU, self).__init__()
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
    def __init__(self, input_size, num_channels, gru_hidden_size, gru_num_layers, output_size):
        super(CNN_GRU, self).__init__()

        # Conv1D Layers
        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=num_channels,
                               kernel_size=3, 
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels,
                               out_channels=num_channels*2,
                               kernel_size=3,
                               padding=1)
        self.pool = nn.MaxPool1d(2)

        # GRU Layers
        self.gru = nn.GRU(input_size=num_channels*2,
                        hidden_size=gru_hidden_size,
                          num_layers=gru_num_layers,
                          batch_first=True,
                          dropout=0.3)

        # Fully Connected Layer
        self.fc = nn.Linear(gru_hidden_size, output_size)


    def forward(self, x):
        # Conv1D Layers
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Convert result of Conv1D fit to inputs of GRU
        x = x.transpose(1, 2)

        # GRU Layer
        out, _ = self.gru(x)

        # Use only the output of the last time step
        out = out[:, -1, :]

        # Fully Connected Layer
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
        self.fc_out = nn.Linear(d_model, 1)  # 가정: targets가 스칼라 값

    def forward(self, src):
        src = self.fc_in(src)  # 입력 차원을 d_model로 변환
        src = src.permute(1, 0, 2)  # Transformer 기대 입력 형태로 차원 순서 변경: (S, N, E)
        output = self.transformer.encoder(src)  # 인코더 한 번만 호출
        output = output.permute(1, 0, 2)  # 원래 배치 차원 순서로 되돌림: (N, S, E)
        output = self.fc_out(output[:, -1, :])  # 마지막 시퀀스 요소의 출력만 사용
        return output