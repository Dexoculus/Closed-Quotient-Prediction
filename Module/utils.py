import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from Module.dataset import wavDataset

def load_data(ft, batch):
    train_group = ['n06', 'n07', 'n08', 'n09', 'n10', 'n14', 'n15', 'n16', 'n17', 'n18', 'n19', 'n20', 'n21', 'n22', 'n23']
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