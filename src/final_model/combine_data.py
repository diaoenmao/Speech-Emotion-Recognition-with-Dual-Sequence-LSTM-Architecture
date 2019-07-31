import pickle
import numpy as np

def check_consistent(hand_dict, spec_512_dict, spec_1024_dict):
    flag = True
    for i in range(len(hand_dict["target"])):
        if np.argmax(hand_dict["target"][i]) == np.argmax(spec_512_dict["target"][i]) == np.argmax(spec_1024_dict["target"][i]):
            continue
        else:
            flag = False
            break
    if flag:
        print("Datasets consistent")
    else:
        raise ValueError("Datasets inconsistent")

def combine(session, option):
    if option == 'full':
        with open('/scratch/speech/final_dataset/IEMOCAP_full.pkl', 'rb') as f:
            hand_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(512) + '_full.pkl', 'rb') as f:
            spec_512_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(1024) + '_full.pkl', 'rb') as f:
            spec_1024_dict = pickle.load(f)
        check_consistent(hand_dict, spec_512_dict, spec_1024_dict)
        dict = {'input_lstm': hand_dict['input'], 'input1': spec_512_dict['input'], 'input2': spec_1024_dict['input'], 'target': hand_dict['target']}
        with open('/scratch/speech/final_dataset/EMO39_mel_spectrogram_full.pkl', 'wb') as full:
            pickle.dump(dict, full)

    elif option == 'train':
        with open('/scratch/speech/final_dataset/IEMOCAP_train_{}.pkl'.format(session), 'rb') as f:
            hand_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(512) + '_train_{}.pkl'.format(session), 'rb') as f:
            spec_512_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(1024) + '_train_{}.pkl'.format(session), 'rb') as f:
            spec_1024_dict = pickle.load(f)
        check_consistent(hand_dict, spec_512_dict, spec_1024_dict)
        dict = {'input_lstm': hand_dict['input'], 'input1': spec_512_dict['input'], 'input2': spec_1024_dict['input'], 'target': hand_dict['target']}
        with open('/scratch/speech/hand_raw_dataset/EMO39_mel_spectrogram_train_{}.pkl'.format(session), 'wb') as train:
            pickle.dump(dict, train)

    elif option == 'test':
        with open('/scratch/speech/final_dataset/IEMOCAP_test_{}.pkl'.format(session), 'rb') as f:
            hand_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(512) + '_train_{}.pkl'.format(session), 'rb') as f:
            spec_512_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(1024) + '_train_{}.pkl'.format(session), 'rb') as f:
            spec_1024_dict = pickle.load(f)
        check_consistent(hand_dict, spec_512_dict, spec_1024_dict)
        dict = {'input_lstm': hand_dict['input'], 'input1': spec_512_dict['input'], 'input2': spec_1024_dict['input'], 'target': hand_dict['target']}
        with open('/scratch/speech/hand_raw_dataset/EMO39_mel_spectrogram_test_{}.pkl'.format(session), 'wb') as test:
            pickle.dump(dict, test)

if __name__ == '__main__':
    combine(0, 'full')
    for i in range(5):
        combine(i, 'train')
        combine(i, 'test')
