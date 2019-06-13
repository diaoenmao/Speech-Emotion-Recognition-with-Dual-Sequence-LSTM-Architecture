import pickle
import numpy as np

pickle_in = open('/scratch/speech/IEMOCAP_dictionary_5.pkl', 'rb')
data = pickle.load(pickle_in)

input = data["input"]
target = data["target"]
seq_length = data["seq_length"]

rand = np.random.randint(0, len(input), int(0.8 * len(input)))

train_input = input[rand]
train_target = target[rand]
train_seq_length = seq_length[rand]

def removeArray(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

for x in train_input:
    input = input.tolist()
    #if x in input:
    #    input.remove(x)
    removeArray(input, x)
    input = np.array(input)

for y in train_target:
    target = target.tolist()
    target.remove(y)
    target = np.array(target)

for z in train_seq_length:
    seq_length = seq_length.tolist()
    seq_length.remove(z)
    seq_length = np.array(seq_length)

#test_input = np.setdiff1d(input, train_input)
#test_target = np.setdiff1d(target, train_target)
#est_seq_length = np.setdiff1d(seq_length, train_seq_length)

train_sample = {"input": train_input, "target": train_target, "seq_length": train_seq_length}
test_sample = {"input": input, "target": target, "seq_length": seq_length}

with open('/scratch/speech/IEMOCAP_dictionary_5_train.pkl', 'wb') as f:
    pickle.dump(train_sample, f)

with open('/scratch/speech/IEMOCAP_dictionary_5_test.pkl', 'wb') as f:
    pickle.dump(test_sample, f)
