#!/bin/bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source $HOME/vllm-env/bin/activate

export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="12.0"
export CMAKE_BUILD_PARALLEL_LEVEL=8

echo "=== Zyphra vLLM Fork Build ===" | tee /tmp/vllm_build.log
echo "Start: $(date)" | tee -a /tmp/vllm_build.log
echo "CUDA: $(nvcc --version 2>/dev/null | tail -1)" | tee -a /tmp/vllm_build.log

pip install setuptools_scm --quiet 2>/dev/null || true
pip install --no-cache-dir "vllm @ git+https://github.com/Zyphra/vllm.git@zaya1-pr" 2>&1 | tee -a /tmp/vllm_build.log

echo "" | tee -a /tmp/vllm_build.log
echo "End: $(date)" | tee -a /tmp/vllm_build.log
python3 -c "import vllm; print('BUILD SUCCESS:', vllm.__version__)" 2>&1 | tee -a /tmp/vllm_build.log
