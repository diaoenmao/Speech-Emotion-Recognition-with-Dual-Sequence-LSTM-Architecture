import os
import numpy as np

OPENSMILE_CONFIG_PATH = '/scratch/speech/opensmile/opensmile-2.3.0/config/MFCC12_E_D_A.conf'
out_file = '/scratch/speech/Speech-Emotion-Analysis/src/preprocessing/temp.csv'

in_file = '/scratch/speech/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav'
if __name__ == '__main__':
	cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -csvoutput ' + out_file + ' -headercsv 0'
	os.system(cmd)
 
