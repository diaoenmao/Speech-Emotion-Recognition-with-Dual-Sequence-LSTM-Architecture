#!/bin/bash
python feature_extraction_emos.py -dn IEMOCAP_PROSODIC_FOUR_EMO -cf /scratch/speech/opensmile/opensmile-2.3.0/config/prosodyShs.conf
python feature_extraction_emos.py -dn IEMOCAP_39_FOUR_EMO -cf /scratch/speech/opensmile/opensmile-2.3.0/config/MFCC12_0_D_A_Z.conf

