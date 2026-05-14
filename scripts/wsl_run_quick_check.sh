#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
/home/ttimm/vllm-env/bin/python3 "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/scripts/wsl_quick_check.py" 2>&1
