import numpy as np
import matplotlib.pyplot as plt
import pickle
import librosa
import pdb

save_path = '/scratch/speech/raw_audio_dataset/'
f = open('/scratch/speech/raw_audio_dataset/raw_audio_full.pkl', 'rb')
data = pickle.load(f)

sample_rate = 16000
nfft = pow(2, 9)

def create_data(data, n_fft):
    utterances_new = []
    for i, utterance in enumerate(data['input']):
        plt.clf()
        mel_spectrogram = librosa.feature.melspectrogram(y=utterance.astype('float'), sr=sample_rate, n_fft=n_fft, hop_length=nfft//2, n_mels=n_fft//4)
        utterances_new.append(mel_spectrogram)
        if i % 1000 == 0:
            print(i)
    print('Parsing through data done.')
    return utterances_new

if __name__ == '__main__':
    utterances_new = create_data(data, n_fft=nfft)
    dataset = {'input': utterances_new, 'target': data['target']}
    with open(save_path + 'mel_spectrogram_nfft{}_updated'.format(nfft) + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)
