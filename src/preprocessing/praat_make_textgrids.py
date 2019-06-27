import os
import pandas as pd

in_file = '/scratch/speech/raw_audio_dataset/audio_paths_labels_updated.csv'
df = pd.read_csv(in_file)

for file, _ in df.values:
    script_path = '/scratch/speech/modularProsodyTagger/mod01.praat'
    index = file.rfind('/')
    basename = file[(index + 1):-4]
    directory = file[:60] + basename[:-5] + '/'
    endpoint = '/scratch/speech/textgrids/'
    cmd = 'praat --run {} {} {} {}'.format(script_path, directory, basename, endpoint)
    os.system(cmd)
