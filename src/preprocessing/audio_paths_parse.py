import pandas as pd

in_file = './audio_paths_labels.csv'
out_file = './audio_paths_labels_updated.csv'

df = pd.read_csv(in_file)

emotions = ['hap', 'exc', 'sad', 'ang', 'neu']
audio_paths = []
categorical_emotion = []
for file, emotion, valence, activation, dominance in df.values:
    if emotion in emotions:
        audio_paths.append(file)
        categorical_emotion.append(emotion)

df_updated = pd.DataFrame(list(zip(audio_paths, categorical_emotion)), columns=['file', 'emotion'])
df_updated.to_csv(out_file, index=False)
