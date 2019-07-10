import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import cv2
from sklearn.model_selection import train_test_split
import pickle
import concurrent.futures
import pdb

in_file = '/scratch/speech/raw_audio_dataset/audio_paths_labels_updated.csv'
df = pd.read_csv(in_file)

encode = {"hap": [1, 0, 0, 0], "exc": [1, 0, 0, 0], "neu": [0, 1, 0, 0], "ang": [0, 0, 1, 0], "sad": [0, 0, 0, 1]}

endpoint = '/scratch/speech/spectrograms_segmented_dpi10_step40_overlap/'

save_path = '/scratch/speech/raw_audio_dataset/'

def create_data(df_value):
    file, emotion = df_value
    sample_rate, sample = wavfile.read(file)
    segments = np.array_split(sample, 40)
    previous=[]
    segments_new=[]
    for s in segments:
        previous+=s
        segments_new.append(previous)
    assert len(previous)==len(sample) , "size mismatch"
    utterance = []
    for j in range(40):
        plt.clf()
        spectrum, freqs, t, im = plt.specgram(segments_new, Fs=sample_rate)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(ticker.NullLocator())
        index = file.rfind('/')
        basename = file[(index + 1):-4]
        plt.savefig(endpoint + '{}_spec_{}.png'.format(basename, j), dpi=10, bbox_inches='tight', pad_inches=0)
        im = cv2.imread(endpoint + '{}_spec_{}.png'.format(basename, j))
        utterance.append(im)
    label = encode[emotion]
    return utterance, label

#def extract_features(dataframe):
#    for i, (file, emotion) in enumerate(dataframe.values):
#        sample_rate, sample = wavfile.read(file)
#        segments = np.array_split(sample, 20)
#        for j, segment in enumerate(segments):
#            spectrum, freqs, t, im = plt.specgram(segment, Fs=sample_rate)
#            plt.gca().set_axis_off()
#            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#            plt.margins(0,0)
#            plt.gca().xaxis.set_major_locator(ticker.NullLocator())
#            plt.gca().yaxis.set_major_locator(ticker.NullLocator())
#            #plt.show()
#            index = file.rfind('/')
#            basename = file[(index + 1):-4]
#            plt.savefig(endpoint + '{}_spec_{}.png'.format(basename, j), bbox_inches='tight', pad_inches=0)
#            print(i, j)
#            im = cv2.imread(endpoint + '{}_spec_{}.png'.format(basename, j))
#            utterance.append(im)
#        input.append(utterance)
#        utterance = []
#        target.append(encode[emotion])
#
#    return input, target

def split_data(data):
    input_train, input_test, target_train, target_test = train_test_split(
        data['input'], data['target'], test_size=0.2, random_state=42)
    train = {'input': input_train, 'target': target_train}
    test = {'input': input_test, 'target': target_test}
    return train, test

def save(dataset):
    train, test = split_data(dataset)
    with open(save_path + 'spectrogram_segmented_dpi10_step40_overlap' + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(save_path + 'spectrogram_segmented_dpi10_step40_overlap' + '_train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(save_path + 'spectrogram_segmented_dpi10_step40_overlap' + '_test.pkl', 'wb') as f:
        pickle.dump(test, f)



if __name__ == '__main__':
    data = []
    input = []
    target = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, (utterance, label) in zip(range(df.shape[0]), executor.map(create_data, df.values)):
            print('Succcessfully generated spectrograms for file #' + str(i))
            data.append((utterance, label))

    for i in data:
        input.append(i[0])
        target.append(i[1])
    dataset = {'input': input, 'target': target}
    save(dataset)
