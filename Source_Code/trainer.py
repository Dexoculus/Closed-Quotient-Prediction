# Import Modules
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from models import *
from dataset import *

# Get device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_data(ft, batch):
    train_group = ['n06', 'n07', 'n08',\
                   'n09', 'n10', 'n14', 'n15', 'n16', 'n17', 'n18',\
                   'n19', 'n20', 'n21', 'n22', 'n23']
    test_group =  ['n01', 'n11', 'n12', 'n13']
    valid_group = ['n03', 'n04', 'n05']

    wav = r'data path'
    csv = r'txt path'

    train_dataset = wavDataset(wav_dir=wav,
                                csv_path=csv,
                                group=train_group,
                                feature=ft) # Train data

    valid_dataset = wavDataset(wav_dir=wav,
                                csv_path=csv,
                                group=valid_group,
                                feature=ft) # Test data
    
    test_dataset = wavDataset(wav_dir=wav,
                                csv_path=csv,
                                group=test_group,
                                feature=ft) # Test data

    train_loader = DataLoader(train_dataset,
                            batch_size=batch,
                            shuffle=True)

    valid_loader = DataLoader(valid_dataset,
                            batch_size=batch,
                            shuffle=True)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=batch,
                            shuffle=True)
    
    return train_loader, valid_loader, test_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mean_absolute_error(outputs, targets):
    return torch.mean(torch.abs(outputs - targets))

def root_mean_squared_error(outputs, targets):
    return torch.sqrt(torch.mean((outputs - targets) ** 2))

def MAPELoss(outputs, targets):
    diff = torch.abs((outputs - targets) / (targets)) * 10
    loss = torch.mean(diff)

    return loss

def plot_performance(title, y_true, y_pred):
    lim = np.array([y_true.min(), y_true.max(), y_pred.min(), y_pred.max()])
    x = np.arange(lim.min(), lim.max()+0.1, 0.1)
    y = x
    
    titledict = {'fontsize': 20,
                 'style': 'normal', # 'oblique' 'italic'
                 'fontweight': 'normal'} # 'bold', 'heavy', 'light', 'ultrabold', 'ultralight

    labeldict = {'fontsize': 15,
                 'style': 'normal', # 'oblique' 'italic'
                 'fontweight': 'normal'} # 'bold', 'heavy', 'light', 'ultrabold', 'ultralight'
    
    x_label = "Exact data"; y_label = "Predict data"
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, c='black')
    plt.plot(x, y, linestyle='-.', color='red')
    plt.xlim(lim.min(), lim.max())
    plt.ylim(lim.min(), lim.max())
    plt.title(title, **titledict)
    plt.xlabel(f'{x_label:>60}', **labeldict)
    plt.ylabel(f'{y_label:>60}', **labeldict)
    plt.show()

from torch.nn.functional import l1_loss  # For MAE calculation

def train_model(model, num_epochs, criterion, optimizer, batch, feature, tag):
    train_loader, valid_loader, test_loader = load_data(feature, batch)

    for epoch in range(num_epochs):
        model.train()  # Training Mode
        total_train_loss = 0
        total_train_mae = 0
        total_train_samples = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mae_loss = l1_loss(outputs, targets)  # MAE calculation

            total_train_loss += loss.item() * inputs.size(0)
            total_train_mae += mae_loss.item() * inputs.size(0)
            total_train_samples += inputs.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_train_loss = total_train_loss / total_train_samples
        average_train_mae = total_train_mae / total_train_samples
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_train_loss:.6f}, Training MAE: {average_train_mae:.6f}')

        model.eval()  # Inference Mode
        total_test_loss = 0
        total_test_mae = 0
        total_test_samples = 0

        with torch.no_grad():
            for inputs, targets in test_loader:  # Test
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mae_loss = l1_loss(outputs, targets)  # MAE calculation

                total_test_loss += loss.item() * inputs.size(0)
                total_test_mae += mae_loss.item() * inputs.size(0)
                total_test_samples += inputs.size(0)

            average_test_loss = total_test_loss / total_test_samples
            average_test_mae = total_test_mae / total_test_samples
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {average_test_loss:.6f}, Test MAE: {average_test_mae:.6f}')

    # Epoch End
    model.eval()  # Inference Mode
    y_exact_test = []
    y_pred_test = []
    y_exact_train = []
    y_pred_train = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:  # Test
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            y_exact_test.extend(targets.cpu().numpy())
            y_pred_test.extend(outputs.cpu().numpy())

        for inputs, targets in train_loader:  # Train
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            y_exact_train.extend(targets.cpu().numpy())
            y_pred_train.extend(outputs.cpu().numpy())

    plot_performance('Train', np.array(y_exact_train), np.array(y_pred_train))
    plot_performance('Test', np.array(y_exact_test), np.array(y_pred_test))


def train_transformer_model(model, num_epochs, criterion, optimizer, batch, feature, tag):
    train_loader, test_loader = load_data(feature, batch)

    for epoch in range(num_epochs):
        model.train() # Training Mode
        total_train_loss = 0
        total_train_samples = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.squeeze(2).to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)            
            total_train_loss += loss.item() * inputs.size(0)
            total_train_samples += inputs.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_train_loss = total_train_loss / total_train_samples
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_train_loss:.6f}')

        model.eval() # Inference Mode
        total_test_loss = 0
        total_test_samples = 0

        with torch.no_grad():

            for inputs, targets in test_loader: # Test
                inputs = inputs.squeeze(2).to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item() * inputs.size(0)
                total_test_samples += inputs.size(0)

            average_test_loss = total_test_loss / total_test_samples
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {average_test_loss:.6f}')
    
    # Epoch End
    model.eval() # Inference Mode
    y_exact_test = []
    y_pred_test = []
    y_exact_train = []
    y_pred_train = []
    with torch.no_grad():
        for inputs, targets in train_loader: # Train
            inputs = inputs.squeeze(2).to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            for i in targets.cpu():
                y_exact_train.append(i.item())
            for i in outputs.cpu():
                y_pred_train.append(i.item())

        for inputs, targets in test_loader: # Test
            inputs = inputs.squeeze(2).to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            for i in targets.cpu():
                y_exact_test.append(i.item())
            for i in outputs.cpu():
                y_pred_test.append(i.item())

    plot_performance('Test', y_exact_test, y_pred_test)
    plot_performance('Train', y_exact_train, y_pred_train)

def train_XGB(max_depth, mcw, eta, gamma, sub, bytree, lam, alpha, seed, rounds, feature, batch=1):
    train_loader, test_loader = load_data(feature, batch)

    # Load Training data
    X_train = []; Y_train = []
    for inputs, targets in train_loader:
        X_train.append(inputs.numpy())
        Y_train.append(targets.numpy())
    X_train = np.array(X_train); Y_train = np.array(Y_train)
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = Y_train.reshape(X_train.shape[0], -1)

    # Load Test data
    X_test = []; Y_test = []
    for inputs, targets in test_loader:
        X_test.append(inputs.numpy())
        Y_test.append(targets.numpy())
    X_test = np.array(X_test); Y_test = np.array(Y_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = Y_test.reshape(X_test.shape[0], -1)

    # Generate DMatrix
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': max_depth,
        'min_child_weight': mcw,
        'gamma': gamma,
        'subsample': sub,
        'colsample_bytree': bytree,
        'lambda': lam,
        'alpha': alpha,
        'eta': eta,
        'eval_metric': 'rmse',
        'seed': seed
    }

    evals = [(dtrain, 'train'), (dtest, 'test')]
    bst = xgb.train(params, dtrain, rounds, evals=evals)

    y_pred_train = bst.predict(dtrain)
    y_exact_train = dtrain.get_label()
    print(len(y_exact_train))

    y_pred_test = bst.predict(dtest)
    y_exact_test = dtest.get_label()
    print(len(y_exact_test))

    plot_performance('Test', y_exact_test, y_pred_test)
    plot_performance('Train', y_exact_train, y_pred_train)

def train_RandomForest(n_estimators, max_depth, feature, batch=1):
    train_loader, test_loader = load_data(feature, batch)

    # Load Training data
    X_train = []; Y_train = []
    for inputs, targets in train_loader:
        X_train.append(inputs.numpy())
        Y_train.append(targets.numpy())
    X_train = np.array(X_train); Y_train = np.array(Y_train)
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = Y_train.reshape(X_train.shape[0], -1)

    # Load Test data
    X_test = []; Y_test = []
    for inputs, targets in test_loader:
        X_test.append(inputs.numpy())
        Y_test.append(targets.numpy())
    X_test = np.array(X_test); Y_test = np.array(Y_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = Y_test.reshape(X_test.shape[0], -1)

    reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    reg.fit(X_train, Y_train)

    y_pred_test = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    print(mean_squared_error(Y_test, y_pred_test))

    plot_performance('Test', Y_test, y_pred_test)
    plot_performance('Train', Y_train, y_pred_train)