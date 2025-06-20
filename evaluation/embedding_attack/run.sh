#!/bin/bash
set +e
cd embedding_eval
source .env
echo "Installing packages"
pip install litellm
pip install trl==0.10.1
pip install transformers==4.44.0

python embedding_attack.py "$@"
