#!/bin/bash

cd /home/ismail/dl/abstraction

rm activations/$1/*
python3 main.py -a $1 --resume checkpoints/$1/checkpoint_$2.tar --evaluate --collect-acts --activation-dir activations/$1/
python3 representation.py -a activations/$1 -m pwcca -n 5000 -r $3 -e $2
