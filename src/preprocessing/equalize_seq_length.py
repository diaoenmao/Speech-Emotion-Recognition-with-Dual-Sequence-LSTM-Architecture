import pickle
import numpy as np
import librosa
import pandas as pd
from raw_audio_create_dict import split_data

dict_path = '/scratch/speech/raw_audio_dataset/raw_audio_full.pkl'
file = open(dict_path, 'rb')
data = pickle.load(file)

audio_path = '/scratch/speech/raw_audio_dataset/audio_paths_labels_updated.csv'
df = pd.read_csv(audio_path)

thresh = 128000
sr_standard = 16000
input_new = []
seq_length_new = []

for i, utterance in enumerate(data['input']):
    if len(utterance) < thresh:
        utterance_new = np.tile(utterance, thresh // len(utterance) + 1)
        utterance_new = utterance_new[0:thresh]
        print(len(utterance_new))
        input_new.append(utterance_new)
        seq_length_new.append(len(utterance_new))
    elif len(utterance) > thresh:
        utterance_new, sr = librosa.load(df['file'][i], sr = thresh/(len(utterance)/sr_standard))
        utterance_new = utterance_new[0:thresh]
        print(len(utterance_new))
        input_new.append(utterance_new)
        seq_length_new.append(len(utterance_new))
    else:
        input_new.append(utterance)
        seq_length_new.append(len(utterance))
    print(i)

dataset_updated = {'input': np.array(input_new), 'target': np.array(data['target']), 'seq_length': seq_length_new}

full_set_new = '/scratch/speech/raw_audio_dataset/raw_audio_full_equal_lengths.pkl'
train_set_new = '/scratch/speech/raw_audio_dataset/raw_audio_train_equal_lengths.pkl'
test_set_new = '/scratch/speech/raw_audio_dataset/raw_audio_test_equal_lengths.pkl'

train, test = split_data(dataset_updated)
with open(full_set_new, 'wb') as f:
    pickle.dump(dataset_updated, f)
with open(train_set_new, 'wb') as f:
    pickle.dump(train, f)
with open(test_set_new, 'wb') as f:
    pickle.dump(test, f)
