#!/bin/bash
sudo apt-get install -y curl mpg123
sudo apt-get install -y mecab mecab-ipadic-utf8 libmecab-dev
sudo apt-get install -y python3.9 python3.9-venv
python3.9 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install mecab-python3
pip install flask pydub
pip install networkx==2.5.0
pip install redis
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import nltk; nltk.download('punkt_tab')"
pip install TTS
#git clone https://github.com/coqui-ai/TTS.git
#cd TTS
#pip install -e .[all]
