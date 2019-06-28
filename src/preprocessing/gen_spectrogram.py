import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pandas as pd

in_file = '/scratch/speech/raw_audio_dataset/audio_paths_labels_updated.csv'
df = pd.read_csv(in_file)

sample_rate, samples = wavfile.read(df.values[0][0])
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
