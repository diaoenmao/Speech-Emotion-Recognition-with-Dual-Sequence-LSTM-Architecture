import os
import pandas as pd
import numpy as np
import pickle
import pdb

OPENSMILE_CONFIG_PATH = '/scratch/speech/opensmile/opensmile-2.3.0/config/MFCC12_E_D_A.conf'
DATASET_PATH = '/scratch/speech/final_dataset/'
dataset_name = 'IEMOCAP/'

out_file = '/scratch/speech/temp.csv'

data = pd.read_csv('/scratch/speech/raw_audio_dataset/audio_paths_labels_updated.csv')

session_indices = [0]

def extract_features(dataframe):
    input = []
    target = []
    for i, (file, emotion) in enumerate(dataframe.values):
        if i % 1000 == 0: print(i)
        if file[44] != dataframe.values[i+1][0][44]:
            pdb.set_trace()
            session_indices.append(i+1)
        cmd = 'SMILExtract -C {} -I {} -csvoutput {} -headercsv 0'.format(OPENSMILE_CONFIG_PATH, file, out_file)
        os.system(cmd)
        df = pd.read_csv(out_file, delimiter=';').iloc[:, 2:]
        input.append(df.values)
        target.append(emotion)
    print(session_indices)
    return input, target

'''
def extract_features_ts(args, dataframe):
    input, target = extract_features(args, dataframe)
    seq_length = [x.shape[0] for x in input]
    return input, target, seq_length
'''

def split_data(data, session):
    input_test = data['input'][session_indices[session]:session_indices[session + 1]]
    target_test = data['target'][session_indices[session]:session_indices[session + 1]]
    input_train = data['input'][:session_indices[session]] + data['input'][session_indices[session + 1]:]
    target_train = data['target'][:session_indices[session]] + data['target'][session_indices[session + 1]:]
    train = {'input': input_train, 'target': target_train}
    test = {'input': input_test, 'target': target_test}
    return train, test

def save(dataset):
    with open(DATASET_PATH + dataset_name + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    for i in range(5):
        train, test = split_data(dataset, i)
        with open(DATASET_PATH + dataset_name + '_train_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(train, f)
        with open(DATASET_PATH + dataset_name + '_test_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(test, f)

if __name__ == '__main__':
    input, target = extract_features(data)
    dataset = {'input': input, 'target': target}
    save(dataset)
