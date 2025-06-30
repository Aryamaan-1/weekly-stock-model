#!/usr/bin/env bash
set -euo pipefail

cd /home/ec2-user/weekly-stock-model

if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install -r model/requirements.txt

python model/model_script.py

deactivate
rm -rf venv

sudo /sbin/shutdown -h now
