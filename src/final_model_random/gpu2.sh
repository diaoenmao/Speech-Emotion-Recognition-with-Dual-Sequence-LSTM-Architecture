#!/bin/bash
python train.py -b 128 -m "model10" -kc1 5 -kc2 5 -f "new_experiment_model10"  -e 5 -g 1 -lr 0.0001
python train.py -b 128 -m "model10" -kc1 4 -kc2 4 -f "new_experiment_model10"  -e 5 -g 2 -lr 0.0001
python train.py -b 128 -m "model10" -kc1 3 -kc2 3 -f "new_experiment_model10"  -e 5 -g 0 -lr 0.0001