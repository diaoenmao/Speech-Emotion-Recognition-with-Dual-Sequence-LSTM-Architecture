#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 100 -nl 4 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 100 -nl 5 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 50 -nl 8 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200 -nl 4 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200 -nl 5 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 300 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.0005 -e 50 -hd 200 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.6 -b 256 -lr 0.001 -e 50 -hd 200 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.6 -b 256 -lr 0.001 -e 50 -hd 200 -tc -no_train -no_test
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.6 -b 256 -lr 0.001 -e 50 -bi -hd 200 -tc -no_train -no_test
