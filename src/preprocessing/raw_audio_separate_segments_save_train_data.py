import pickle
import pdb

# train set changes
path = '/scratch/speech/raw_audio_dataset/raw_audio_segmented_train.pkl'
out = open(path, 'rb')
data = pickle.load(out)

input = []
target = []
for i, utterance in enumerate(data['input']):
    for j, segment in enumerate(utterance):
        if data['segment_labels'][i][j] != 'SIL':
            input.extend(segment)
            target.append(data['target'][i])

train_dataset = {'input': input, 'target': target}
pdb.set_trace()
with open(path, 'wb') as f:
    pickle.dump(train_dataset, f)

# test set changes
path = '/scratch/speech/raw_audio_dataset/raw_audio_segmented_test.pkl'
out = open(path, 'rb')
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
with open(path, 'wb') as f:
    pickle.dump(test_dataset, f)
