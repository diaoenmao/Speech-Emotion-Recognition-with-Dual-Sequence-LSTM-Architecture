#!/bin/bash
python train.py -m "model4" -f "experiment_model4"
python train.py -out1 64 -out2 32 -kc1 5 -kc2 3 -w 0.55 -f "experiment_model4" -m "model4"
python train.py -out1 16 -out2 64 -kc1 4 -kc2 4 -w 0.6 -f "experiment_model4" -m "model4"
python train.py -out1 32 -out2 64 -kc1 4 -kc2 2 -w 0.5 -f "experiment_model4" -m "model4"
python train.py -out1 32 -out2 64 -kc1 5 -kc2 3 -w 0.5 -f "experiment_model4" -m "model4"
python train.py -out1 64 -out2 32 -kc1 5 -kc2 3 -w 0.6 -f "experiment_model4" -m "model4"
python train.py -out1 64 -out2 32 -kc1 4 -kc2 4 -w 0.5 -f "experiment_model4" -m "model4"
python train.py -out1 128 -out2 16 -kc1 3 -kc2 3 -w 0.5 -f "experiment_model4" -m "model4"
