import pickle
import numpy as np
import pdb

dataset_path = '/scratch/speech/final_dataset_random/'
f = open('/scratch/speech/final_dataset/EMO39_mel_spectrogram_full.pkl', 'rb')
data = pickle.load(f)

input_lstm = data['input_lstm']
input1 = data['input1']
input2 = data['input2']
target = data['target']

p = np.random.permutation(len(input_lstm)).tolist()
input_lstm = [input_lstm[i] for i in p]
input1 = [input1[i] for i in p]
input2 = [input2[i] for i in p]
target = [target[i] for i in p]
dict = {'input_lstm': input_lstm, 'input1': input1, 'input2': input2, 'target': target}

def equal_chunks(l, i):
    return [l[:i], l[i:(2*i)], l[(2*i):(3*i)], l[(3*i):(4*i)], l[(4*i):]]

def check_consistent(hand_dict, spec_512_dict, spec_1024_dict):
    flag = True
    for i in range(len(hand_dict["target"])):
        if np.argmax(hand_dict["target"][i]) == np.argmax(spec_512_dict["target"][i]) == np.argmax(spec_1024_dict["target"][i]):
            continue
        else:
            flag = False
            print(i)
            break
    if flag:
        print("Datasets consistent")
    else:
        raise ValueError("Datasets inconsistent")

def split_data(data, fold):
    interval = len(input_lstm) // 5
    x = [equal_chunks(data[key], interval) for key in data.keys()]

    input_lstm_test = x[0][fold]
    input1_test = x[1][fold]
    input2_test = x[2][fold]
    target_test = x[3][fold]

    x[0].remove(input_lstm_test)
    x[1].remove(input1_test)
    x[2].remove(input2_test)
    x[3].remove(target_test)

    input_lstm_train = np.concatenate(x[0]).tolist()
    input1_train = np.concatenate(x[1]).tolist()
    input2_train = np.concatenate(x[2]).tolist()
    target_train = np.concatenate(x[3]).tolist()

    train = {'input_lstm': input_lstm_train, 'input1': input1_train, 'input2': input2_train, 'target': target_train}
    test = {'input_lstm': input_lstm_test.tolist(), 'input1': input1_test.tolist(), 'input2': input2_test.tolist(), 'target': target_test.tolist()}

def create_pickle(dict, fold):
    train, test = split_data(dict, fold)
    with open(dataset_path + 'EMO39_mel_spectrogram_train_{}.pkl'.format(fold), 'wb') as f:
        pickle.dump(train, f)
    with open(dataset_path + 'EMO39_mel_spectrogram_test_{}.pkl'.format(fold), 'wb') as f:
        pickle.dump(test, f)

if __name__ == '__main__':
    with open(dataset_path + 'EMO39_mel_spectrogram_full.pkl', 'wb') as full:
        pickle.dump(dict, full)
    for i in range(5):
        create_pickle(dict, i)
