#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -bi -hd 200
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -dr 0.2 -b 256 -lr 0.001 -e 50 -bi -hd 200
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.4 -b 256 -lr 0.001 -e 50 -bi -hd 200
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.6 -b 256 -lr 0.001 -e 50 -bi -hd 200
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -dr 0.8 -b 256 -lr 0.001 -e 50 -bi -hd 200
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -dr 0.2 -b 256 -lr 0.001 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.4 -b 256 -lr 0.001 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.6 -b 256 -lr 0.001 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -dr 0.8 -b 256 -lr 0.001 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -dr 0.2 -b 256 -lr 0.001 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.4 -b 256 -lr 0.001 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.6 -b 256 -lr 0.001 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="3" python full_train_and_test.py -dr 0.8 -b 256 -lr 0.001 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.01 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.0005 -e 50 -hd 200
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 250
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 300
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200 --nl 3
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200 -nl 4
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 200 -nl 5
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 100 --nl 3
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 100 -nl 4
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 100 -nl 5
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 50 --nl 6
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 50 -nl 7
CUDA_VISIBLE_DEVICES="2" python full_train_and_test.py -dr 0.0 -b 256 -lr 0.001 -e 50 -hd 50 -nl 8
CUDA_VISIBLE_DEVICES="0" python full_train_and_test.py -dr 0.2 -b 256 -lr 0.001 -e 20 -bi -hd 250
CUDA_VISIBLE_DEVICES="1" python full_train_and_test.py -dr 0.2 -b 256 -lr 0.001 -e 20 -bi -hd 250