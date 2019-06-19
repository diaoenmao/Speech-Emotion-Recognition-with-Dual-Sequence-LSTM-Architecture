#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.6 -b 256 -lr 0.0005 -e 50 -hd 200 -bi
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.2 -b 256 -lr 0.0005 -e 50 -hd 250 -bi
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.6 -b 256 -lr 0.0005 -e 50 -hd 250 -bi
