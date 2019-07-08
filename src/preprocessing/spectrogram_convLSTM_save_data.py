import pickle
from sklearn.model_selection import train_test_split

file = open('/scratch/speech/raw_audio_dataset/spectrogram_full.pkl', 'rb')
data = pickle.load(file)

step = 10

pixel_group = []
row_new = []
spectrogram_ts = []
input_new = []

def extract_features():
    for spectrogram in data['input']:
        for row in spectrogram:
            for i, pixel in enumerate(row):
                if i % step != 0:
                    pixel_group.append(pixel)
                else:
                    row_new.append(pixel_group)
                    pixel_group = []
            spectrogram_ts.append(row_new)
            row_new = []
        input_new.append(spectrogram_ts)
        spectrogram_ts = []
    return input_new, data['target']

def split_data(data):
    input_train, input_test, target_train, target_test = train_test_split(
        data["input"], data["target"], test_size=0.2, random_state=42)
    train = {'input': input_train, 'target': target_train}
    test = {'input': input_test, 'target': target_test}
    return train, test

def save(dataset):
    train, test = split_data(dataset)
    with open(save_path + 'spectrogram_ts' + '_full.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(save_path + 'spectrogram_ts' + '_train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(save_path + 'spectrogram_ts' + '_test.pkl', 'wb') as f:
        pickle.dump(test, f)

if __name__ == '__main__':
    input, target = extract_features()
    dataset = {'input': input, 'target': target}
    save(dataset)
