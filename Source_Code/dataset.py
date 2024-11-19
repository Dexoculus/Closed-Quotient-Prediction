import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Functions for Data preprocessing

# Normalize Audio data
def Amplitude_Normalization(y):
    audio_data = []
    max_signal = max(y); min_signal = min(y)
    for i, signal in enumerate(y):
        norm_signal = (signal - min_signal)/(max_signal - min_signal)
        audio_data.append(norm_signal)
    audio_data = np.array(audio_data)

    return audio_data

# Raw data
def _Extract_audio(wav_dir, file_df):
    audio_data = []
    for filename in file_df['filename']:
        path = os.path.join(wav_dir, filename)
        y, sr = librosa.load(path, mono=False)
        signal = Amplitude_Normalization(y[0])
        audio_data.append(signal.T)

    return audio_data

# Get Pitches from audio
def _audio_pitch(wav_dir, file_df):
    pitches = []
    for filename in file_df['filename']:
        path = os.path.join(wav_dir, filename)
        y, sr = librosa.load(path, mono=False)
        signal = Amplitude_Normalization(y[0])
        pitch, _ = librosa.piptrack(y=signal, sr=sr)
        pitch = pitch.mean(axis=0)
        pitches.append(pitch)
        
    return pitches

# Get Chroma Features from audio
def _audio_chroma(wav_dir, file_df):
    chromas = []
    for filename in file_df['filename']:
        path = os.path.join(wav_dir, filename)
        y, sr = librosa.load(path, mono=False)
        signal = Amplitude_Normalization(y[0])
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        chromas.append(chroma.T)

    return chromas

# Get Mel-Frequency Cepstral Coefficient from audio
def _audio_MFCC(wav_dir, file_df):
    audio_data = []
    for filename in file_df['filename']:
        path = os.path.join(wav_dir, filename)
        y, sr = librosa.load(path, mono=False)
        signal = Amplitude_Normalization(y[0])
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        audio_data.append(mfcc.T)  

    return audio_data

# Get Zero Crossing Rate from audio
def _audio_ZCR(wav_dir, file_df):
    zcrs = []
    for filename in file_df['filename']:
        path = os.path.join(wav_dir, filename)
        y, sr = librosa.load(path, mono=False)
        signal = Amplitude_Normalization(y[0])
        zcr = librosa.feature.zero_crossing_rate(signal)
        zcrs.append(zcr.T)

    return zcrs

# Get RMS Energy from audio
def _audio_energy(wav_dir, file_df):
    energies = []
    for filename in file_df['filename']:
        path = os.path.join(wav_dir, filename)
        y, sr = librosa.load(path, mono=False)
        signal = Amplitude_Normalization(y[0])
        energy = librosa.feature.rms(y=signal)
        energies.append(energy.T)

    return energies

def _spectrogram(wav_dir, file_df):
    spec = []
    for filename in file_df['filename']:
        path = os.path.join(wav_dir, filename)
        y, sr = librosa.load(path, mono=False)
        s = np.abs(librosa.stft(y[0]))
        S_db = librosa.amplitude_to_db(s, ref=np.max)
        spec.append(S_db)

    return spec

def normalize_CQ(df):
    list_CQ = []
    for i, y in enumerate(df):
        y_min = min(df); y_max = max(df)
        n_CQ = (y - y_min) / (y_max - y_min)
        list_CQ.append(n_CQ)

    return list_CQ

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
                self.signal = _Extract_audio(path, df)
            elif(feature == 'Pitch'):
                self.signal = _audio_pitch(path, df)
            elif(feature == 'MFCC'):
                self.signal = _audio_MFCC(path, df)
            elif(feature == 'Chroma'):
                self.signal = _audio_chroma(path, df)
            elif(feature == 'ZCR'):
                self.singal = _audio_ZCR(path, df)
            elif(feature == "Energy"):
                self.signal = _audio_energy(path, df)
            elif(feature == 'Spectrogram'):
                self.signal = _spectrogram(path, df)
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

# Function for padding sequence
def my_collate_fn(batch):
    inputs, outputs = zip(*batch)
    padded_inputs = pad_sequence(inputs,
                                 batch_first=True,
                                 padding_value=0)
    
    outputs = torch.stack(outputs)

    return padded_inputs, outputs