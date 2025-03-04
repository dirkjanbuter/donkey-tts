FROM ubuntu:22.04

WORKDIR /usr/local/app

RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    libsndfile1 \
    mecab \
    mecab-ipadic-utf8 \
    libmecab-dev \
    python3.10 \
    python3.10-venv

COPY requirements.txt ./
COPY install.docker ./install.sh

RUN chmod u+x ./install.sh
RUN bash ./install.sh

COPY main.py ./
COPY run.sh ./
RUN chmod u+x ./run.sh

EXPOSE 8979

CMD ["bash", "run.sh"]
