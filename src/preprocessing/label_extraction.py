import os
import numpy as np
import pandas as pd
import wave

PATH_TO_DATA = '/scratch/speech/IEMOCAP_full_release/'


def print_list(list_to_print):
	for item in list_to_print:
		print(item)


def process_text_files():
	audio_paths = []
	categorical_emotion = []
	average_valence = []
	average_activation = []
	average_dominance = []
	for x in range(5):
		session_path = PATH_TO_DATA + 'Session' + str(x + 1)
		label_path = session_path + '/dialog/EmoEvaluation'
		audio_path = session_path + '/sentences/wav/'
		files = [label_path + '/' + f for f in os.listdir(label_path) if
				 os.path.isfile(os.path.join(label_path, f)) and f[0] != '.']
		# print_list(files)
		for file in files:
			lines = []
			get_next_line = False
			with open(file) as f:
				for line in f:
					if get_next_line and len(line.strip('\n').split('\t')) > 3:
						lines.append(line.strip().split('\t'))
					elif line.strip() == '':
						get_next_line = True
			for utterance in lines:
				audio_paths.append(audio_path + utterance[1][:14] + '/' + utterance[1] + '.wav')
				categorical_emotion.append(utterance[2])
				print(utterance)
				scores = utterance[3][1:-1].split(', ')
				average_valence.append(float(scores[0]))
				average_activation.append(float(scores[1]))
				average_dominance.append(float(scores[2]))
	return pd.DataFrame(
		list(zip(audio_paths, categorical_emotion, average_valence, average_activation, average_dominance)),
		columns=['file', 'emotion', 'valence', 'activation', 'dominance'])


if __name__ == '__main__':
	df = process_text_files()
	df.to_csv('/scratch/rpc21/Speech-Emotion-Analysis/src/preprocessing/audio_paths_labels.csv',index=False)
	# print('wrote to csv')
