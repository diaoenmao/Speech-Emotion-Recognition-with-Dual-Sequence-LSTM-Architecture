import pickle
import numpy as np
import random

file = open('/scratch/speech/raw_audio_dataset/raw_audio_train.pkl', 'rb')
data = pickle.load(file)

path = '/scratch/speech/raw_audio_dataset/'
sr = 16000

happy = []
neutral = []
angry = []
sad = []

input = data['input']
target = data['target']
seq_length = data['seq_length']

for utterance, label in zip(data['input'], data['target']):
    if label == [1,0,0,0]:
        happy.append(utterance)
    elif label == [0,1,0,0]:
        neutral.append(utterance)
    elif label == [0,0,1,0]:
        angry.append(utterance)
    else:
        sad.append(utterance)

emotions = [happy, neutral, angry, sad]
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

    for y, utterance in enumerate(emotion):
        print("Sample " + str(y) + " from emotion " + str(x))
        if len(utterance)/sr < 8:
            if len(utterance)/sr > 7:
                utterances_7_8.append(utterance)
            elif len(utterance)/sr > 6:
                utterances_6_7.append(utterance)
            elif len(utterance)/sr > 5:
                utterances_5_6.append(utterance)
            elif len(utterance)/sr > 4:
                utterances_4_5.append(utterance)
            elif len(utterance)/sr > 3:
                utterances_3_4.append(utterance)
            elif len(utterance)/sr > 2:
                utterances_2_3.append(utterance)
            elif len(utterance)/sr > 1:
                utterances_1_2.append(utterance)
            else:
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

    rand_0_8 = np.random.choice(matrix_0_8, size=2000)
    for (i, j) in rand_0_8:
        additions.append(np.append(utterances_0_1[i], utterances_7_8[j]))
        a = np.zeros(4)
        np.put(a, x, 1)
        target.append(a)
        seq_length.append(len(utterances_0_1[i] + len(utterances_7_8[j])))

    rand_1_7 = np.random.choice(matrix_1_7, size=2000)
    for (i, j) in rand_1_7:
        additions.append(np.append(utterances_1_2[i], utterances_6_7[j]))
        a = np.zeros(4)
        np.put(a, x, 1)
        target.append(a)
        seq_length.append(len(utterances_1_2[i] + len(utterances_6_7[j])))

    rand_2_6 = np.random.choice(matrix_2_6, size=2000)
    for (i, j) in rand_2_6:
        additions.append(np.append(utterances_2_3[i], utterances_5_6[j]))
        a = np.zeros(4)
        np.put(a, x, 1)
        target.append(a)
        seq_length.append(len(utterances_2_3[i] + len(utterances_5_6[j])))

    rand_3_5 = np.random.choice(matrix_3_5, size=2000)
    for (i, j) in rand_3_5:
        additions.append(np.append(utterances_3_4[i], utterances_4_5[j]))
        a = np.zeros(4)
        np.put(a, x, 1)
        target.append(a)
        seq_length.append(len(utterances_3_4[i] + len(utterances_4_5[j])))

input += additions
if len(input) == len(target) == len(seq_length):
    print("equal lengths")
x = list(range(len(input)))
random.shuffle(x)
input_new = []
target_new = []
seq_length_new = []
for i in x:
    input_new.append(input[i])
    target_new.append(target[i])
    seq_length_new.append(seq_length[i])

dict = {'input': input_new, 'target': target_new, 'seq_length_new': seq_length_new}
with open(path + 'raw_audio_augmented' + '_train.pkl', 'wb') as f:
    pickle.dump(dict, f)
