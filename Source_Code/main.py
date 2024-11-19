from trainer import *

def main_RNN():
    tag = "model"
    feature_list = ['Raw', 'Pitch', 'MFCC', 'Chroma', 'ZCR', 'Energy', 'Spectrogram']
    learning_rate = 0.005

    model = model_GRU(input_size=1,
                        hidden_size=32,
                        num_layers=2,
                        output_size=1).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    feature = feature_list[1]
    num_epochs = 200
    batch = 1
    train_model(model, num_epochs, criterion, optimizer, batch, feature, tag)

    total_params = count_parameters(model)
    print(f'Total Parameters: {total_params}')

def main_Transformer():
    tag = "model"
    feature_list = ['Raw', 'Pitch', 'MFCC', 'Chroma', 'ZCR', 'Energy', 'Spectrogram']
    learning_rate = 0.005

    model = Transformer(input_dim=5,
                        d_model=128,
                        nhead=8,
                        num_encoder_layers=3,
                        num_decoder_layers=3,
                        dim_feedforward=512).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    feature = feature_list[6]
    num_epochs = 100
    batch = 1
    train_transformer_model(model, num_epochs, criterion, optimizer, batch, feature, tag)

    total_params = count_parameters(model)
    print(f'Total Parameters: {total_params}')

def main_XGB():
    feature_list = ['Raw', 'Pitch', 'MFCC', 'Chroma', 'ZCR', 'Energy', 'Spectrogram']
    train_XGB(max_depth=4,
              mcw=2,
              eta=0.05,
              gamma=0.2,
              sub=0.8,
              bytree=0.8,
              lam=1,
              alpha=0,
              seed=42,
              rounds=1000,
              feature=feature_list[2])

def main_forest():
    feature_list = ['Raw', 'Pitch', 'MFCC', 'Chroma', 'ZCR', 'Energy', 'Spectrogram']
    train_RandomForest(
        n_estimators=400,
        max_depth=80,
        feature=feature_list[2]
    )

def main():
    main_RNN()

if __name__ == "__main__":
    main()