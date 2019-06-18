#!/bin/bash
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -dr 0.8 -b 256 -lr 0.001 -e 50 -bi -hd 200 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -dr 0.8 -b 256 -lr 0.001 -e 50 -hd 200 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -dr 0.8 -b 256 -lr 0.001 -e 50 -hd 200 -tc -no_train -no_test
