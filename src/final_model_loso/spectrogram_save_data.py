from scipy.io import wavfile
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import librosa

df = pd.read_csv('/scratch/speech/raw_audio_dataset/audio_paths_labels_updated.csv')
DATASET_PATH = '/scratch/speech/final_dataset/'

encode = {"hap": [1, 0, 0, 0], "exc": [1, 0, 0, 0], "neu": [0, 1, 0, 0], "ang": [0, 0, 1, 0], "sad": [0, 0, 0, 1]}

session_indices = [0]

def extract_data(dataframe, nfft):
    input = []
    target = []
    for i, (file, emotion) in enumerate(dataframe.values):
        if i % 500 == 0: print(i)
        if i < len(dataframe.values) - 1 and file[44] != dataframe.values[i+1][0][44]:
            session_indices.append(i+1)
        sample_rate, data = wavfile.read(file)
        plt.clf()
        mel_spectrogram = librosa.feature.melspectrogram(y=data.astype('float'), sr=sample_rate, n_fft=nfft, hop_length=nfft//2, n_mels=nfft//4)
        input.append(mel_spectrogram)
        target.append(encode[emotion])
    return input, target

def split_data(data, session):
    if session < len(session_indices) - 1:
        input_test = data['input'][session_indices[session]:session_indices[session + 1]]
        target_test = data['target'][session_indices[session]:session_indices[session + 1]]
        input_train = data['input'][:session_indices[session]] + data['input'][session_indices[session + 1]:]
        target_train = data['target'][:session_indices[session]] + data['target'][session_indices[session + 1]:]
    else:
        input_test = data['input'][session_indices[session]:]
        target_test = data['target'][session_indices[session]:]
        input_train = data['input'][:session_indices[session]]
        target_train = data['target'][:session_indices[session]]
    train = {'input': input_train, 'target': target_train}
    test = {'input': input_test, 'target': target_test}
    return train, test

def save(dataset, nfft):
    with open(DATASET_PATH + 'mel_spectrogram_nfft{}'.format(nfft) + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    for i in range(5):
        train, test = split_data(dataset, i)
        with open(DATASET_PATH + 'mel_spectrogram_nfft{}'.format(nfft)+ '_train_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(train, f)
        with open(DATASET_PATH + 'mel_spectrogram_nfft{}'.format(nfft)+ '_test_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(test, f)

if __name__ == '__main__':
    nfft = 1024
    input, target = extract_data(df, nfft)
    dataset = {'input': input, 'target': target}
    save(dataset, nfft)
