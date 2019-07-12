import numpy as np
import matplotlib.pyplot as plt
import pickle
import librosa
import pdb

save_path = '/scratch/speech/raw_audio_dataset/'
infile = open('/scratch/speech/raw_audio_dataset/raw_audio_full.pkl', 'rb')
data = pickle.load(infile)

sample_rate = 16000
nfft = pow(2, 8)

def create_data(data):
    utterances_new = []
    utterances_new_mel = []
    for i, utterance in enumerate(data['input']):
        plt.clf()
        lin_spectrogram, freqs, t, im = plt.specgram(utterance, Fs=sample_rate, NFFT=nfft)
        mel_spectrogram = librosa.feature.melspectrogram(segment.astype('float'), sr=sample_rate, n_fft=nfft)
        utterances_new.append(lin_spectrogram)
        utterances_new_mel.append(mel_spectrogram)
        print(i)
    return utterances_new, utterances_new_mel, data['target']

def save(linear_dataset, mel_dataset):
    with open(save_path + 'linear_spectrogram_nfft{}'.format(nfft) + '_full.pkl', 'wb') as f:
        pickle.dump(linear_dataset, f)
    with open(save_path + 'mel_spectrogram_nfft{}'.format(nfft) + '_full.pkl', 'wb') as f:
        pickle.dump(mel_dataset, f)

if __name__ == '__main__':
    utterances_new, utterances_new_mel, target = create_data(data)
    linear_dataset = {'input': utterances_new, 'target': target}
    mel_dataset = {'input': utterances_new_mel, 'target': target}
    save(linear_dataset, mel_dataset)
