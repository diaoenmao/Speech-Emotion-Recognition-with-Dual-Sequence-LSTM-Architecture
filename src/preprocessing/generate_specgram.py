import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

in_file = '/scratch/speech/raw_audio_dataset/audio_paths_labels_updated.csv'
df = pd.read_csv(in_file)

endpoint = '/scratch/speech/spectrograms/'

for file, _ in df.values:
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
    plt.savefig(endpoint + '{}_spec.png'.format(basename), bbox_inches='tight', pad_inches=0)
