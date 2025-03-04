#!/bin/bash
sudo apt-get install -y curl mpg123
sudo apt-get install -y mecab mecab-ipadic-utf8 libmecab-dev
sudo apt-get install -y python3.10 python3.10-venv
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
