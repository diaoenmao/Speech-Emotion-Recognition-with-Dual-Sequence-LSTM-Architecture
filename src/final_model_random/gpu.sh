#!/bin/bash
python train.py -m "model4" -kc1 5 -kc2 5 -f "new_experiment_model4"  -e 10
python train.py -m "model4" -kc1 4 -kc2 4 -f "new_experiment_model4" -e 10
python train.py -m "model4" -kc1 3 -kc2 3 -f "new_experiment_model4" -e 10
python train.py -out1 64 -out2 32 -kc1 5 -kc2 5 -w 0.5 -f "new_experiment_model4" -m "model4" -e 10
python train.py -out1 64 -out2 32 -kc1 4 -kc2 4 -w 0.5 -f "new_experiment_model4" -m "model4" -e 10
python train.py -out1 64 -out2 32 -kc1 3 -kc2 3 -w 0.5 -f "new_experiment_model4" -m "model4" -e 10
python train.py -out1 128 -out2 16 -kc1 5 -kc2 5 -w 0.5 -f "new_experiment_model4" -m "model4" -e 10
python train.py -out1 128 -out2 16 -kc1 4 -kc2 4 -w 0.5 -f "new_experiment_model4" -m "model4" -e 10
python train.py -out1 128 -out2 16 -kc1 3 -kc2 3 -w 0.5 -f "new_experiment_model4" -m "model4" -e 10
