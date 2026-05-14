#!/bin/bash
# Install Zyphra transformers fork for zaya model support
set -e
source /home/ttimm/vllm-env/bin/activate
echo "Installing Zyphra transformers fork..."
pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya1" 2>&1 | tail -10
echo "Done"
pip show transformers | grep -E 'Name|Version|Location'
