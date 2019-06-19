#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -m lstm -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200 -bi -tc
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -m lstm -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200 -tc
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -m lstm -dr 0.2 -b 256 -lr 0.001 -e 50 -hd 200 -tc
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -m lstm -dr 0.4 -b 256 -lr 0.001 -e 50 -hd 200 -tc
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -m lstm -dr 0.3 -b 256 -lr 0.001 -e 50 -hd 200 -tc
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -m lstm -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 250 -tc
