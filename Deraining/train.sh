#!/usr/bin/env bash

CONFIG=$1

python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch