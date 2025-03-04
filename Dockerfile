FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

WORKDIR /usr/local/app

COPY requirements.txt ./

RUN apt-get update
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8979

RUN useradd app
USER app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8979"]
