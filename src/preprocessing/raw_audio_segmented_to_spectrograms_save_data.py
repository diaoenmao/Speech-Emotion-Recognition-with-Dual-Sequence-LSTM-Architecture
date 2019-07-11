import numpy as np
import matplotlib.pyplot as plt
import pickle
import librosa
import pdb

save_path = '/scratch/speech/raw_audio_dataset/'
infile = open('/scratch/speech/raw_audio_dataset/raw_audio_segmented_full.pkl', 'rb')
data = pickle.load(infile)

sample_rate = 16000

def create_data(data):
    segments_new = []
    segments_new_mel = []
    utterances_new = []
    utterances_new_mel = []
    for i, utterance in enumerate(data['input']):
        for j, segment in enumerate(utterance):
            lin_spectrogram, freqs, t, im = plt.specgram(segment, Fs=sample_rate)
            segments_new.append(lin_spectrogram)
            mel_spectrogram = librosa.feature.melspectrogram(segment, sr=sample_rate)
            segments_new_mel.append(mel_spectrogram)
        utterances_new.append(segments_new)
        utterances_new_mel.append(segments_new_mel)
        segments_new = []
        segments_new_mel = []
        print(i)
    return utterances_new, utterances_new_mel, data['target'], data['segment_labels']

def save(linear_dataset, mel_dataset):
    with open(save_path + 'linear_spectrogram_segmented' + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(save_path + 'mel_spectrogram_segmented' + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    utterances_new, utterances_new_mel, target, segment_labels = create_data(data)
    linear_dataset = {'input': utterances_new, 'target': target, 'segment_labels': segment_labels}
    mel_dataset = {'input': utterances_new_mel, 'target': target, 'segment_labels': segment_labels}
    save(linear_dataset, mel_dataset)
