import pickle
import numpy as np

pickle_in = open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_full.pkl', 'rb')
data = pickle.load(pickle_in)
new_list = []
index = []
encode = {"hap": [1, 0, 0, 0], "neu": [0, 1, 0, 0], "ang": [0, 0, 1, 0], "sad": [0, 0, 0, 1]}
for i, t in enumerate(data["target"]):
    try:
        new_list.append(encode[t])
        index.append(i)
    except:
        continue
new_input = data["input"][index]
new_seq_length = data["seq_length"][index]
sample = {"input": new_input, "seq_length": new_seq_length, "target": np.array(new_list)}
with open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_full.pkl', 'wb') as f:
    pickle.dump(sample, f)
