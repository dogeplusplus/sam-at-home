#!/bin/bash
set -xeuf -o pipefail

rm -rf venv
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

pip install git+https://github.com/facebookresearch/segment-anything.git
