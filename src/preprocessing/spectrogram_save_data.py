import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import cv2
from sklearn.model_selection import train_test_split

in_file = '/scratch/speech/raw_audio_dataset/audio_paths_labels_updated.csv'
df = pd.read_csv(in_file)

encode = {"hap": [1, 0, 0, 0], "exc": [1, 0, 0, 0], "neu": [0, 1, 0, 0], "ang": [0, 0, 1, 0], "sad": [0, 0, 0, 1]}

endpoint = '/scratch/speech/spectrograms/'

input = []
target = []

save_path = '/scratch/speech/raw_audio_dataset/'

def extract_features(dataframe):
    for i, (file, emotion) in enumerate(dataframe.values):
        sample_rate, samples = wavfile.read(file)
        spectrum, freqs, t, im = plt.specgram(samples, Fs=sample_rate)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(ticker.NullLocator())
        #plt.show()
        index = file.rfind('/')
        basename = file[(index + 1):-4]
        #plt.savefig(endpoint + '{}_spec.png'.format(basename), bbox_inches='tight', pad_inches=0)
        #print(i)
        im = cv2.imread(endpoint + '{}_spec.png'.format(basename))
        input.append(im)
        target.append(encode[emotion])

    return input, target

def split_data(data):
    input_train, input_test, target_train, target_test = train_test_split(
        data["input"], data["target"], test_size=0.2, random_state=42)
    train = {'input': input_train, 'target': target_train}
    test = {'input': input_test, 'target': target_test}
    return train, test

def save(dataset):
    train, test = split_data(dataset)
    with open(save_path + 'spectrogram' + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(save_path + 'spectrogram' + '_train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(save_path + 'spectrogram' + '_test.pkl', 'wb') as f:
        pickle.dump(test, f)

if __name__ == '__main__':
    input, target = extract_features(df)
    dataset = {'input': input, 'target': target}
    save(dataset)
