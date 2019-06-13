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

train_sample = {"input": train_input, "target": train_target, "seq_length": train_seq_length}
test_sample = {"input": input, "target": target, "seq_length": seq_length}

with open('/scratch/speech/IEMOCAP_dictionary_5_train.pkl', 'wb') as f:
    pickle.dump(train_sample, f)

with open('/scratch/speech/IEMOCAP_dictionary_5_test.pkl', 'wb') as f:
    pickle.dump(test_sample, f)
