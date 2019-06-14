import pickle
import numpy as np
from numpy import *

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
    #else:
    #    raise ValueError('array not found in list.')

def removeList(L, list):
    ind = 0
    size = len(L)
    while ind != size and L[ind] != list:
        ind += 1
    if ind != size:
        L.pop(ind)

#input = map(tuple, input)
#train_input2 = map(tuple, train_input)
#input = set(input)
#train_input2 = set(train_input2)
#test_input = input - train_input2
#test_input = np.array(list(test_input))

input = input.tolist()
for x in train_input:
    #if x in input:
        #input.remove(x)
    removeArray(input, x)
    #input = map(tuple, input)
    #input = set(input)

input = np.array(input)

target = target.tolist()
for y in train_target:
    removeList(target, y) #error is removing a list from a list, same as above ### FIX!

target = np.array(target)

seq_length = seq_length.tolist()

for z in train_seq_length:
    removeList(seq_length, z)

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
