from fastapi import FastAPI, HTTPException, UploadFile, File, Response, Request, Form
from pydantic import BaseModel, ValidationError
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
import torch
import io
import soundfile as sf
import os
import traceback
from pydub import AudioSegment
import json
import os
from typing import Optional
import numpy as np
import logging
import re
import asyncio
from fastapi.responses import StreamingResponse
import wave
from transformers import AutoTokenizer
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a directory to store speaker voice samples if it doesn't exist
SPEAKER_DIR = "speakers"
os.makedirs(SPEAKER_DIR, exist_ok=True)

app = FastAPI()

# Load the model
try:
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
    config = XttsConfig()
    config.load_json("model/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="model/", eval=True)
    model.cuda()
    tokenizer = model.tokenizer
    print("Model succesvol geladen met XTTS klasse.")
except Exception as e:
    print(f"Fout bij het laden van het model: {e}")
    raise HTTPException(status_code=500, detail=f"Model laden mislukt: {str(e)}")

class TTSRequest(BaseModel):
    text: str
    language: str

def split_text_into_paragraphs_and_sentences(text):
    paragraphs = text.split('\n\n')
    result = []
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    for paragraph in paragraphs:
        sentences = sentence_endings.split(paragraph)
        result.append(sentences)
    return result

def chunk_text(text, language, max_tokens=250, tokenizer=None):
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None")
    tokens = tokenizer.encode(text, lang=language)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

async def generate_audio_stream(text, language, speaker_wav_path, tokenizer=None, model=None, config=None):
    try:
        max_text_length = config.model_args.gpt_max_text_tokens #Get the max text length from the config.
        inputs = tokenizer.encode(text, lang=language)

        # Pad the input sequence
        padding_length = max_text_length - len(inputs)
        if padding_length > 0:
            inputs = inputs + [tokenizer.pad_token_id] * padding_length
        elif padding_length < 0:
            inputs = inputs[:max_text_length] #truncate the input if it is too long.

        input_ids = torch.tensor(inputs).unsqueeze(0).cuda()

        # Create the attention mask
        attention_mask = torch.ones_like(input_ids).cuda()

        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Speaker wav file path: {speaker_wav_path}")
        audio, sr = librosa.load(speaker_wav_path, sr=48000)
        print(f"Speaker wav file shape: {audio.shape}, sample rate: {sr}")

        outputs = model.synthesize(
            text,
            config,
            speaker_wav=speaker_wav_path,
            language=language,
            attention_mask=attention_mask
        )
        # ... rest of your code ...
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}", exc_info=True)
        yield b""

def overlap_add(audio_segments, overlap_samples=200):
    if not audio_segments:
        return None  # Return None if audio_segments is empty

    combined_audio = np.concatenate(audio_segments)
    result = np.zeros_like(combined_audio)
    hop_length = len(audio_segments[0]) - overlap_samples
    window = librosa.filters.get_window("hann", overlap_samples * 2)

    for i, segment in enumerate(audio_segments):
        if segment.size == 0: # controleer of segment niet leeg is.
            continue
        start = i * hop_length
        end = start + len(segment)

        if i > 0:
            overlap_start = start
            overlap_end = overlap_start + overlap_samples
            result[overlap_start:overlap_end] += segment[:overlap_samples] * window[:overlap_samples]

        result[start + (overlap_samples if i > 0 else 0):end] += segment[overlap_samples if i > 0 else 0:]

    return result

@app.post("/tts_stream/")
async def text_to_speech_stream(
    text: str = Form(...),
    language: str = Form(...),
    speaker_id: str = Form(...),
    req: Request = None
):
    try:
        print(f"Received TTS streaming request for speaker: {speaker_id}, language: {language}")
        speaker_wav_path = os.path.join(SPEAKER_DIR, f"{speaker_id}.wav")
        if not os.path.exists(speaker_wav_path):
            raise HTTPException(status_code=404, detail=f"Speaker ID '{speaker_id}' not found")

        audio_stream = generate_audio_stream(text, language, speaker_wav_path, tokenizer=tokenizer)
        return StreamingResponse(audio_stream, media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"Error in TTS streaming processing: {e}", exc_info=True)
        return Response(
            content=json.dumps({"detail": str(e)}),
            media_type="application/json",
            status_code=500,
        )

@app.get("/list_speakers/")
async def list_speakers():
    try:
        speakers = []
        for filename in os.listdir(SPEAKER_DIR):
            if filename.endswith(".wav"):
                speaker_id = filename[:-4]
                speakers.append(speaker_id)
        return {"speakers": speakers}
    except Exception as e:
        logger.error(f"Error listing speakers: {e}", exc_info=True)
        return Response(
            content=json.dumps({"detail": "Ophalen van sprekerlijst mislukt"}),
            media_type="application/json",
            status_code=500,
        )

@app.post("/upload_speaker/")
async def upload_speaker(speaker_id: str = Form(...), file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are allowed.")
        speaker_path = os.path.join(SPEAKER_DIR, f"{speaker_id}.wav")
        with open(speaker_path, "wb") as buffer:
            buffer.write(await file.read())
        return {"message": f"Speaker '{speaker_id}' uploaded successfully."}
    except Exception as e:
        logger.error(f"Error uploading speaker: {e}", exc_info=True)
        return Response(content=json.dumps({"detail": str(e)}), media_type="application/json", status_code=500)
