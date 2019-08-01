#!/bin/bash
python train_FT_LSTM4.py -b 128 -f "experiment6"
python train_FT_LSTM4.py -b 128 -out1 64 -out2 32 -kc1 5 -kc2 3 -sc1 1 -sc2 1 -kp1 2 -kp2 2 -sp 2 -w 0.55 -f "experiment6"
python train_FT_LSTM4.py -b 128 -out1 16 -out2 64 -kc1 4 -kc2 4 -sc1 1 -sc2 1 -kp1 2 -kp2 2 -sp 2 -w 0.6 -f "experiment6"
python train_FT_LSTM4.py -b 128 -out1 32 -out2 64 -kc1 4 -kc2 2 -sc1 1 -sc2 1 -kp1 2 -kp2 2 -sp 2 -w 0.5 -f "experiment6"
python train_FT_LSTM4.py -b 128 -out1 32 -out2 64 -kc1 5 -kc2 3 -sc1 1 -sc2 1 -kp1 2 -kp2 2 -sp 2 -w 0.5 -f "experiment6"
python train_FT_LSTM4.py -b 128 -out1 64 -out2 32 -kc1 5 -kc2 3 -sc1 1 -sc2 1 -kp1 2 -kp2 2 -sp 2 -w 0.6 -f "experiment6"
python train_FT_LSTM4.py -b 128 -out1 64 -out2 32 -kc1 4 -kc2 4 -sc1 1 -sc2 1 -kp1 2 -kp2 2 -sp 2 -w 0.5 -f "experiment6"
python train_FT_LSTM4.py -b 128 -out1 128 -out2 16 -kc1 3 -kc2 3 -sc1 1 -sc2 1 -kp1 2 -kp2 2 -sp 2 -w 0.5 -f "experiment6"