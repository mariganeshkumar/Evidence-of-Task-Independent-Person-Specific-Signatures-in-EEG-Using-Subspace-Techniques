#!/usr/bin/env bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 
export CUDA_VISIBLE_DEVICES="$5"
python library/xvector/scripts/Train_And_Test_Mod_Xvectors_Cosine.py $1 $2 $3 $4
