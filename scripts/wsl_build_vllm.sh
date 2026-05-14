#!/bin/bash
export PATH=/usr/local/cuda/bin:/usr/bin:$PATH
source ~/vllm-env/bin/activate
pip install --force-reinstall 'vllm @ git+https://github.com/Zyphra/vllm.git@zaya1-pr' 2>&1 | tee /tmp/vllm_build.log
echo "BUILD DONE"
pip show vllm | grep Version
