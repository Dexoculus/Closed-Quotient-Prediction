import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from Module.preprocessing import *

# Custom dataset -> [batch, seq_length, input_size]
class wavDataset(Dataset):
    def __init__(self, wav_dir, csv_path, group, feature='Raw'):
        self.audio = []
        self.frames = []
        self.CQ = pd.DataFrame(columns=['CQ'])
        for person in group:
            path = os.path.join(wav_dir, person)
            df_path = os.path.join(csv_path, f'{person}.csv')
            df = pd.read_csv(df_path)
            # Extract Audio data
            if(feature == 'Raw'):
                self.signal = Extract_audio(path, df)
            elif(feature == 'Pitch'):
                self.signal = audio_pitch(path, df)
            elif(feature == 'MFCC'):
                self.signal = audio_MFCC(path, df)
            elif(feature == 'Chroma'):
                self.signal = audio_chroma(path, df)
            elif(feature == 'ZCR'):
                self.singal = audio_ZCR(path, df)
            elif(feature == "Energy"):
                self.signal = audio_energy(path, df)
            elif(feature == 'Spectrogram'):
                self.signal = spectrogram(path, df)
            self.audio += self.signal
            # Target data
            self.frames.append(df['CQ'])

        self.CQ = pd.concat(self.frames, ignore_index=True)
        self.data = self.audio
        self.labels = self.CQ

    def __getitem__(self, index):
        audio_tensor = torch.from_numpy(self.data[index]).float()
        audio_tensor = audio_tensor.unsqueeze(1)
        label_tensor = torch.tensor(self.labels[index]).float()

        return audio_tensor, label_tensor
    
    def __len__(self):
        return len(self.data)