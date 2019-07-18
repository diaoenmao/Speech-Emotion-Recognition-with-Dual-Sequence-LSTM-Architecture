import numpy as np
import matplotlib.pyplot as plt
import pickle
import librosa
import pdb

save_path = '/scratch/speech/hand_raw_dataset/'
f_train = open('/scratch/speech/hand_raw_dataset/EMO39_raw_audio_augmented_train.pkl', 'rb')
f_test = open('/scratch/speech/hand_raw_dataset/EMO39_raw_audio_augmented_test.pkl', 'rb')
train_data = pickle.load(f_train)
test_data = pickle.load(f_test)

sample_rate = 16000
nfft = pow(2, 9)

def create_data(train_data, test_data, n_fft):
    utterances_new_train = []
    utterances_new_mel_train = []
    utterances_new_test = []
    utterances_new_mel_test = []

    for i, utterance_train in enumerate(train_data['input']):
        plt.clf()
        lin_spectrogram_train, freqs, t, im = plt.specgram(utterance_train, Fs=sample_rate, NFFT=n_fft)
        mel_spectrogram_train = librosa.feature.melspectrogram(y=utterance_train.astype('float'), sr=sample_rate, n_fft=n_fft)
        #mel_spectrogram = librosa.feature.melspectrogram(S=lin_spectrogram, n_fft=nfft)
        utterances_new_train.append(lin_spectrogram_train)
        utterances_new_mel_train.append(mel_spectrogram_train)
        print(i)
    print('Parsing through train data done.')
    for i, utterance_test in enumerate(test_data['input']):
        plt.clf()
        lin_spectrogram_test, freqs, t, im = plt.specgram(utterance_test, Fs=sample_rate, NFFT=n_fft)
        mel_spectrogram_test = librosa.feature.melspectrogram(y=utterance_test.astype('float'), sr=sample_rate, n_fft=n_fft)
        #mel_spectrogram = librosa.feature.melspectrogram(S=lin_spectrogram, n_fft=nfft)
        utterances_new_test.append(lin_spectrogram_test)
        utterances_new_mel_test.append(mel_spectrogram_test)
        print(i)
    print('Parsing through test data done.')
    return utterances_new_train, utterances_new_mel_train, utterances_new_test, utterances_new_mel_test

if __name__ == '__main__':
    utterances_new_train, utterances_new_mel_train, utterances_new_test, utterances_new_mel_test = create_data(train_data, test_data, n_fft=nfft)

    input_lstm_train = train_data['input_lstm']
    target_train = train_data['target']
    input_lstm_test = test_data['input_lstm']
    target_test = test_data['target']

    linear_dataset_train = {'input_lstm': input_lstm_train, 'input': utterances_new_train, 'target': target_train}
    mel_dataset_train = {'input_lstm': input_lstm_train, 'input': utterances_new_mel_train, 'target': target_train}
    linear_dataset_test = {'input_lstm': input_lstm_test, 'input': utterances_new_test, 'target': target_test}
    mel_dataset_test = {'input_lstm': input_lstm_test, 'input': utterances_new_mel_test, 'target': target_test}

    with open(save_path + 'EMO39_linear_spectrogram_nfft{}_augmented'.format(nfft) + '_train.pkl', 'wb') as f:
        pickle.dump(linear_dataset_train, f)
    with open(save_path + 'EMO39_mel_spectrogram_nfft{}_augmented'.format(nfft) + '_train.pkl', 'wb') as f:
        pickle.dump(mel_dataset_train, f)
    with open(save_path + 'EMO39_linear_spectrogram_nfft{}_augmented'.format(nfft) + '_test.pkl', 'wb') as f:
        pickle.dump(linear_dataset_test, f)
    with open(save_path + 'EMO39_mel_spectrogram_nfft{}_augmented'.format(nfft) + '_test.pkl', 'wb') as f:
        pickle.dump(mel_dataset_test, f)
