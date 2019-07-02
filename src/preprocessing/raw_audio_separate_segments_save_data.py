import pickle
import pdb

path = '/scratch/speech/raw_audio_dataset/'

# train set changes
out = open(path + 'raw_audio_segmented_train_equal_lengths.pkl', 'rb')
data = pickle.load(out)

input = []
target = []
for i, utterance in enumerate(data['input']):
    for j, segment in enumerate(utterance):
        if data['segment_labels'][i][j] != 'SIL':
            input.append(segment)
            target.append(data['target'][i])

train_dataset = {'input': input, 'target': target}
pdb.set_trace()
with open(path + 'raw_audio_separate_segments_train.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)

# test set changes
out = open(path + 'raw_audio_segmented_test_equal_lengths.pkl', 'rb')
data = pickle.load(out)

input = []
for i, utterance in enumerate(data['input']):
    utterance_new = []
    for j, segment in enumerate(utterance):
        if data['segment_labels'][i][j] != 'SIL':
            utterance_new.append(segment)
    input.append(utterance_new)

test_dataset = {'input': input, 'target': data['target']}
pdb.set_trace()
with open(path + 'raw_audio_separate_segments_test.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)
