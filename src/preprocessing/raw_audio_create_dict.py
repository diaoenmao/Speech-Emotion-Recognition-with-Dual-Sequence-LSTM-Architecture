from scipy.io import wavfile
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

path = '/scratch/speech/raw_audio_dataset/'
df = pd.read_csv(path + 'audio_paths_labels_updated.csv')

def extract_features(dataframe):
    input = []
    seq_length = []
    target = []
    for file, emotion in dataframe.values:
        sample_rate, data = wavfile.read(file)
        input.append(data)
        target.append(emotion)
        seq_length.append(shape(data))
    return input, target, seq_length

def split_data(data):
    input_train, input_test, target_train, target_test, seq_length_train, seq_length_test = train_test_split(
        data["input"], data["target"], data["seq_length"], test_size=0.2, random_state=42)
    train = {'input': input_train, 'target': target_train, 'seq_length': seq_length_train}
    test = {'input': input_test, 'target': target_test, 'seq_length': seq_length_test}
    return train, test

def save(dataset):
    train, test = split_data(dataset)
    with open(path + 'raw_audio' + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(path + 'raw_audio' + '_train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(path + 'raw_audio' + '_test.pkl', 'wb') as f:
        pickle.dump(test, f)

if __name__ == '__main__':
    input, target, seq_length = extract_features(df)
    dataset = {'input': input, 'target': target, 'seq_length': seq_length}
    save(dataset)
