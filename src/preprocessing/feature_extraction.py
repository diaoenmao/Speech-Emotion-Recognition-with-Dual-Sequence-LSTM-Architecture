import os
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import pdb

OPENSMILE_CONFIG_PATH = '/scratch/speech/opensmile/opensmile-2.3.0/config/MFCC12_E_D_A.conf'
DATASET_PATH = '/scratch/speech/datasets/'
out_file = '/scratch/speech/Speech-Emotion-Analysis/src/preprocessing/temp.csv'

in_file = '/scratch/speech/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav'


def init_parser():
    parser = argparse.ArgumentParser(
        description='Use this script to extract features using OpenSMILE by specifying the path to the configuration file')
    parser.add_argument('-config_filename', '-cf', default=OPENSMILE_CONFIG_PATH, help='Path to the configuration file',
                        dest='config_path')
    parser.add_argument('-dataset_name', '-dn', help='Specify the dataset name / output file (pkl)',
                        dest='dataset_name')
    parser.add_argument('-time_series', '-ts', default=False, action='store_true',
                        help='Include if extracts time series data', dest='time_series')
    return parser.parse_args()


def load_paths_and_labels():
    return pd.read_csv('/scratch/speech/datasets/audio_paths_labels.csv')


def extract_features(args, dataframe):
    input = []
    target = []
    for file, emotion, valence, activation, dominance in dataframe.values:
        cmd = 'SMILExtract -C {} -I {} -csvoutput {} -headercsv 0'.format(args.config_path, file, out_file)
        os.system(cmd)
        if args.time_series:
            df = pd.read_csv(out_file, delimiter=';').iloc[:, 2:]
        else:
            df = pd.read_csv(out_file, delimiter=';').iloc[:, 1:]
        input.append(df.values)
        target.append(emotion)
    #    input = np.array(input)
    #    target = np.array(target)
    return input, target


def extract_features_ts(args, dataframe):
    input, target = extract_features(args, dataframe)
    seq_length = [x.shape[0] for x in input]
    return input, target, seq_length


def split_data(data, time_series):
    if time_series:
        input_train, input_test, target_train, target_test, seq_length_train, seq_length_test = train_test_split(
            data["input"], data["target"], data["seq_length"], test_size=0.2, random_state=42)
        train = {'input': input_train, 'target': target_train, 'seq_length': seq_length_train}
        test = {'input': input_test, 'target': target_test, 'seq_length': seq_length_test}
        return train, test
    else:
        input_train, input_test, target_train, target_test = train_test_split(
            data["input"], data["target"], test_size=0.2, random_state=42)
        train = {'input': input_train, 'target': target_train}
        test = {'input': input_test, 'target': target_test}
        return train, test


def save(args, dataset):
    train, test = split_data(dataset, args.time_series)
    with open(DATASET_PATH + args.dataset_name + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(DATASET_PATH + args.dataset_name + '_train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(DATASET_PATH + args.dataset_name + '_test.pkl', 'wb') as f:
        pickle.dump(test, f)


if __name__ == '__main__':
    args = init_parser()
    dataframe = load_paths_and_labels()
    if args.time_series:
        input, target, seq_length = extract_features_ts(args, dataframe)
        dataset = {'input': input, 'target': target, 'seq_length': seq_length}
    else:
        input, target = extract_features(args, dataframe)
        dataset = {'input': input, 'target': target}
    save(args, dataset)
