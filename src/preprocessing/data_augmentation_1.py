import pickle
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pdb

def split_data(data):
    input_lstm_train, input_lstm_test, input_train, input_test, target_train, target_test = train_test_split(
        data['input_lstm'], data['input'], data['target'], test_size=0.2, random_state=42)
    train = {'input_lstm': input_lstm_train, 'input': input_train, 'target': target_train}
    test = {'input_lstm': input_lstm_test, 'input': input_test, 'target': target_test}
    return train, test

def augment_data(train_data):
    sr = 16000
    happy = []
    neutral = []
    angry = []
    sad = []

    input_lstm = train_data['input_lstm'].tolist()
    input = train_data['input']
    target = train_data['target']

    for utterance_lstm, utterance, label in zip(input_lstm, input, target):
        if label == [1,0,0,0]:
            happy.append((utterance_lstm, utterance))
        elif label == [0,1,0,0]:
            neutral.append((utterance_lstm, utterance))
        elif label == [0,0,1,0]:
            angry.append((utterance_lstm, utterance))
        else:
            sad.append((utterance_lstm, utterance))

    emotions = [happy, neutral, angry, sad]
    additions_lstm = []
    additions = []

    for x, emotion in enumerate(emotions):

        utterances_0_1 = []
        utterances_1_2 = []
        utterances_2_3 = []
        utterances_3_4 = []
        utterances_4_5 = []
        utterances_5_6 = []
        utterances_6_7 = []
        utterances_7_8 = []

        utterances_0_1_lstm = []
        utterances_1_2_lstm = []
        utterances_2_3_lstm = []
        utterances_3_4_lstm = []
        utterances_4_5_lstm = []
        utterances_5_6_lstm = []
        utterances_6_7_lstm = []
        utterances_7_8_lstm = []

        for y, (utterance_lstm, utterance) in enumerate(emotion):
            print("Sample " + str(y) + " from emotion " + str(x))
            if len(utterance)/sr < 8:
                if len(utterance)/sr > 7:
                    utterances_7_8_lstm.append(utterance_lstm)
                    utterances_7_8.append(utterance)
                elif len(utterance)/sr > 6:
                    utterances_6_7_lstm.append(utterance_lstm)
                    utterances_6_7.append(utterance)
                elif len(utterance)/sr > 5:
                    utterances_5_6_lstm.append(utterance_lstm)
                    utterances_5_6.append(utterance)
                elif len(utterance)/sr > 4:
                    utterances_4_5_lstm.append(utterance_lstm)
                    utterances_4_5.append(utterance)
                elif len(utterance)/sr > 3:
                    utterances_3_4_lstm.append(utterance_lstm)
                    utterances_3_4.append(utterance)
                elif len(utterance)/sr > 2:
                    utterances_2_3_lstm.append(utterance_lstm)
                    utterances_2_3.append(utterance)
                elif len(utterance)/sr > 1:
                    utterances_1_2_lstm.append(utterance_lstm)
                    utterances_1_2.append(utterance)
                else:
                    utterances_0_1_lstm.append(utterance_lstm)
                    utterances_0_1.append(utterance)

        matrix_0_8 = []
        for i in range(len(utterances_0_1)):
            for j in range(len(utterances_7_8)):
                matrix_0_8.append((i, j))

        matrix_1_7 = []
        for i in range(len(utterances_1_2)):
            for j in range(len(utterances_6_7)):
                matrix_1_7.append((i, j))

        matrix_2_6 = []
        for i in range(len(utterances_2_3)):
            for j in range(len(utterances_5_6)):
                matrix_2_6.append((i, j))

        matrix_3_5 = []
        for i in range(len(utterances_3_4)):
            for j in range(len(utterances_4_5)):
                matrix_3_5.append((i, j))

        rand_0_8 = np.random.choice(np.array(matrix_0_8, dtype='i,i'), size=2000)
        for (i, j) in rand_0_8:
            additions_lstm.append(np.append(utterances_0_1_lstm[i], utterances_7_8_lstm[j], axis=0))
            additions.append(np.append(utterances_0_1[i], utterances_7_8[j]))
            a = np.zeros(4)
            np.put(a, x, 1)
            target.append(a)

        rand_1_7 = np.random.choice(np.array(matrix_1_7, dtype='i,i'), size=2000)
        for (i, j) in rand_1_7:
            additions_lstm.append(np.append(utterances_1_2_lstm[i], utterances_6_7_lstm[j], axis=0))
            additions.append(np.append(utterances_1_2[i], utterances_6_7[j]))
            a = np.zeros(4)
            np.put(a, x, 1)
            target.append(a)

        rand_2_6 = np.random.choice(np.array(matrix_2_6, dtype='i,i'), size=2000)
        for (i, j) in rand_2_6:
            additions_lstm.append(np.append(utterances_2_3_lstm[i], utterances_5_6_lstm[j], axis=0))
            additions.append(np.append(utterances_2_3[i], utterances_5_6[j]))
            a = np.zeros(4)
            np.put(a, x, 1)
            target.append(a)

        rand_3_5 = np.random.choice(np.array(matrix_3_5, dtype='i,i'), size=2000)
        for (i, j) in rand_3_5:
            additions_lstm.append(np.append(utterances_3_4_lstm[i], utterances_4_5_lstm[j], axis=0))
            additions.append(np.append(utterances_3_4[i], utterances_4_5[j]))
            a = np.zeros(4)
            np.put(a, x, 1)
            target.append(a)

    input_lstm += additions_lstm
    input += additions
    if len(input_lstm) == len(input) == len(target):
        print("equal lengths")
    x = list(range(len(input_lstm)))
    random.shuffle(x)
    input_lstm_new = []
    input_new = []
    target_new = []
    for i in x:
        input_lstm_new.append(input_lstm[i])
        input_new.append(input[i])
        target_new.append(target[i])
    train_new = {'input_lstm': input_lstm_new, 'input': input_new, 'target': target_new}
    return train_new

def combine():
    with open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_full.pkl', 'rb') as out1:
        dict1 = pickle.load(out1)
    with open('/scratch/speech/raw_audio_dataset/raw_audio_full.pkl', 'rb') as out2:
        dict2 = pickle.load(out2)

    flag = True
    for i in range(len(dict1["target"])):
        if np.argmax(dict1["target"][i]) == np.argmax(dict2["target"][i]):
            continue
        else:
            flag = False
            break
    if flag:
        print("Datasets consistent")
    else:
        raise ValueError("Datasets inconsistent")

    dict3 = {"input_lstm": dict1["input"], "input": dict2["input"], "target": dict2["target"]}
    train, test = split_data(dict3)
    train_new = augment_data(train)

    #with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_full.pkl'.format(nfft), 'wb') as full:
        #pickle.dump(dict3,full)
    with open('/scratch/speech/hand_raw_dataset/EMO39_raw_audio_augmented_train.pkl', 'wb') as f:
        pickle.dump(train_new, f)
    with open('/scratch/speech/hand_raw_dataset/EMO39_raw_audio_augmented_test.pkl', 'wb') as f:
        pickle.dump(test, f)

    print("Successfully copied to pickle.")

if __name__ == '__main__':
    combine()
