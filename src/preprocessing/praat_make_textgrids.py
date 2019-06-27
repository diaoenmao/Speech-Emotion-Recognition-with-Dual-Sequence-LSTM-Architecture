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

def create_textgrids():
    input = []
    target = []
    for file, emotion in df.values:
        script_path = '/scratch/speech/modularProsodyTagger/mod01.praat'
        index = file.rfind('/')
        basename = file[(index + 1):-4]
        directory = file[:60] + basename[:-5] + '/'
        endpoint = '/scratch/speech/textgrids/'
        cmd = 'praat --run {} {} {} {}'.format(script_path, directory, basename, endpoint)
        os.system(cmd)

        tgrid_path = endpoint + basename + '_result.TextGrid'
        tgrid = textgrid.read_textgrid(tgrid_path)
        tgrid_df = pd.DataFrame(tgrid)

        data = []
        indices = []
        sample_rate = 0
        for start, stop, name, tier in tgrid_df.values:
            sample_rate, data = wavfile.read(file)
            if tier != 'silences':
                break
            else:
                indices.append(stop)
        data = data.tolist()
        data = [data[i : j] for i, j in zip([0] + indices, indices + [None])]

        input.append(data)
        target.append(encode[emotion])
