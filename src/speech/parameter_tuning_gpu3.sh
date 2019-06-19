#!/bin/bash
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -m mean_pool_gru -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200 -tc -no_train
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -m mean_pool_gru -dr 0.0 -b 256 -lr 0.0001 -e 50 -hd 200 -tc -no_train
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -m mean_pool_gru -dr 0.2 -b 256 -lr 0.0001 -e 50 -hd 200 -tc -no_train
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -m mean_pool_gru -dr 0.4 -b 256 -lr 0.0001 -e 50 -hd 200 -tc -no_train
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -m mean_pool_gru -dr 0.6 -b 256 -lr 0.0001 -e 50 -hd 200 -tc -no_train
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -m mean_pool_gru -dr 0.8 -b 256 -lr 0.0001 -e 50 -hd 200 -tc -no_train
