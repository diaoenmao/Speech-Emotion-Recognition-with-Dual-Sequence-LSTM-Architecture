#!/bin/bash
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.4 -b 256 -lr 0.001 -e 50 -bi -hd 200 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.4 -b 256 -lr 0.001 -e 50 -hd 200 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.4 -b 256 -lr 0.001 -e 50 -hd 200 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 250 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200 -nl 3 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 100 -nl 3 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 50 -nl 7 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.2 -b 256 -lr 0.001 -e 50 -bi -hd 250 -tc -no_train -no_test
