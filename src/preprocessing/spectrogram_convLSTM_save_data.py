import pickle

file = open('/scratch/speech/raw_audio_dataset/spectrogram_full.pkl', 'rb')
data = pickle.load(file)

step = 10

pixel_group = []
pixel_group_step = []
row_new = []
spectrogram_ts = []
input_new = []
for spectrogram in data['input']:
    for row in spectrogram:
        for i, pixel in enumerate(row):
            if i % step != 0:
                pixel_group.append(pixel)
            else:
                pixel_group_step.append(pixel_group)
                pixel_group = []
        row_new.append(pixel_group_step)
        pixel_group_step = []
        spectrogram_ts.append(row_new)
        row_new = []

input_new.append()
