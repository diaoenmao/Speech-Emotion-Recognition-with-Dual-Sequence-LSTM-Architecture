import os
import pandas as pd
import textgrid
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

in_file = '/scratch/speech/raw_audio_dataset/audio_paths_labels_updated.csv'
df = pd.read_csv(in_file)

basename = ''
directory = ''
endpoint = ''

encode = {"hap": [1, 0, 0, 0], "exc": [1, 0, 0, 0], "neu": [0, 1, 0, 0], "ang": [0, 0, 1, 0], "sad": [0, 0, 0, 1]}

save_path = '/scratch/speech/raw_audio_dataset/'

def extract_features():
    input = []
    target = []

    for file, emotion in df.values:
        script_path = '/scratch/speech/modularProsodyTagger/mod01.praat'
        index = file.rfind('/')
        basename = file[(index + 1):-4]
        directory = file[:60] + basename[:-5] + '/'
        endpoint = '/scratch/speech/textgrids/'
        cmd = 'praat --run {} {} {} {}'.format(script_path, directory, basename, endpoint)
        #os.system(cmd)

        tgrid_path = endpoint + basename + '_result.TextGrid'
        tgrid = textgrid.read_textgrid(tgrid_path)
        tgrid_df = pd.DataFrame(tgrid)

        data = []
        indices = []

        for start, stop, name, tier in tgrid_df.values:
            sample_rate, data = wavfile.read(file)
            if tier != 'silences':
                break
            else:
                indices.append(round(stop * sample_rate))
                
        data = data.tolist()
        data = [data[i : j] for i, j in zip([0] + indices, indices + [None])]
        print(data)

        input.append(data)
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
    with open(save_path + 'raw_audio_segmented' + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(save_path + 'raw_audio_segmented' + '_train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(save_path + 'raw_audio_segmented' + '_test.pkl', 'wb') as f:
        pickle.dump(test, f)

if __name__ == '__main__':
    input, target = extract_features(df)
    dataset = {'input': input, 'target': target}
    save(dataset)
