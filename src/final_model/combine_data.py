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

def combine(name, session, option):
    if option == 'full':
        with open('/scratch/speech/final_dataset/IEMOCAP_full.pkl', 'rb') as f:
            hand_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(512) + '_full.pkl', 'rb') as f:
            spec_512_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(1024) + '_full.pkl', 'rb') as f:
            spec_1024_dict = pickle.load(f)
        check_consistent(hand_dict, spec_512_dict, spec_1024_dict)
    elif option == 'train':
        with open('/scratch/speech/final_dataset/IEMOCAP_train_{}.pkl'.format(session), 'rb') as f:
            hand_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(512) + '_train_{}.pkl'.format(session), 'rb') as f:
            spec_512_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(1024) + '_train_{}.pkl'.format(session), 'rb') as f:
            spec_1024_dict = pickle.load(f)
        check_consistent(hand_dict, spec_512_dict, spec_1024_dict)
    elif option == 'test':
        with open('/scratch/speech/final_dataset/IEMOCAP_test_{}.pkl'.format(session), 'rb') as f:
            hand_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(512) + '_train_{}.pkl'.format(session), 'rb') as f:
            spec_512_dict = pickle.load(f)
        with open('/scratch/speech/final_dataset/mel_spectrogram_nfft{}'.format(1024) + '_train_{}.pkl'.format(session), 'rb') as f:
            spec_1024_dict = pickle.load(f)
        check_consistent(hand_dict, spec_512_dict, spec_1024_dict)


    dict = {'input_lstm': hand_dict['input'], 'input1': spec_512_dict['input']
    train1,test1=split_data(dict3)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_full.pkl', 'wb') as full:
        pickle.dump(dict3,full)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_train.pkl', 'wb') as train:
        pickle.dump(train1,train)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_test.pkl', 'wb') as test:
        pickle.dump(test1,test)
