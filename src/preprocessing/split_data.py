import pickle
from sklearn.model_selection import train_test_split

file = open('./IEMOCAP_dictionary_5.pkl', 'rb')
data = pickle.load(file)

input_train, input_test, target_train, target_test, seq_length_train, seq_length_test =
    train_test_split(data["input"], data["target"], data["seq_length"], test_size=0.2, random_state=42)

print(input_train.shape, input_test.shape, target_train.shape, target_test.shape, seq_length_train.shape, seq_length_test.shape)

train_sample = {"input": input_train, "seq_length": seq_length_train, "target": target_train}
test_sample = {"input": input_test, "seq_length": seq_length_test, "target": target_test}

with open('/scratch/speech/IEMOCAP_dictionary_5_train.pkl', 'wb') as f:
    pickle.dump(train_sample, f)

with open('/scratch/speech/IEMOCAP_dictionary_5_test.pkl', 'wb') as f:
    pickle.dump(test_sample, f)
