import pickle
import numpy as np
import librosa
import pandas as pd
from raw_audio_segmented_save_data import split_data

path = '/scratch/speech/raw_audio_dataset/raw_audio_segmented_full.pkl'
file = open(path, 'rb')
data = pickle.load(file)

thresh = 32000
sr_standard = 16000
input_new = []
#seq_length_new = []

for i, utterance in enumerate(data['input']):
    utterance_new = []
    for j, segment in enumerate(utterance):
        if len(segment) < thresh:
            segment_new = np.tile(segment, thresh // len(segment) + 1)
            segment_new = segment_new[0:thresh]
            utterance_new.append(segment_new)
            print(len(segment_new))
            #seq_length_new.append(len(utterance_new))
        elif len(segment) > thresh:
            segment_new = librosa.resample(segment.astype('float'), sr_standard, thresh/(len(segment)/sr_standard))
            segment_new = segment_new[0:thresh]
            utterance_new.append(segment_new)
            print(len(segment_new))
            #seq_length_new.append(len(utterance_new))
        else:
            utterance_new.append(segment)
            print(len(segment_new))
            #seq_length_new.append(len(utterance))
    input_new.append(utterance_new)

dataset_updated = {'input': input_new, 'target': data['target'], 'segment_labels': data['segment_labels']}

full_set_new = '/scratch/speech/raw_audio_dataset/raw_audio_segmented_full_equal_lengths.pkl'
train_set_new = '/scratch/speech/raw_audio_dataset/raw_audio_segmented_train_equal_lengths.pkl'
test_set_new = '/scratch/speech/raw_audio_dataset/raw_audio_segmented_test_equal_lengths.pkl'

train, test = split_data(dataset_updated)
with open(full_set_new, 'wb') as f:
    pickle.dump(dataset_updated, f)
with open(train_set_new, 'wb') as f:
    pickle.dump(train, f)
with open(test_set_new, 'wb') as f:
    pickle.dump(test, f)
