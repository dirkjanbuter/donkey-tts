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
    tokenizer = model.tokenizer  # Access the tokenizer from the loaded model
    print("Model succesvol geladen met XTTS klasse.")
except Exception as e:
    print(f"Fout bij het laden van het model: {e}")
    raise HTTPException(status_code=500, detail=f"Model laden mislukt: {str(e)}")


class TTSRequest(BaseModel):
    text: str
    language: str


def split_text_into_paragraphs_and_sentences(text):
    """Splits text into paragraphs and then sentences."""
    paragraphs = text.split('\n\n')  # Split by double newlines for paragraphs
    result = []
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    for paragraph in paragraphs:
        sentences = sentence_endings.split(paragraph)
        result.append(sentences)
    return result

def chunk_text(text, language, max_tokens=250, tokenizer = None):
    """Chunks text into segments based on token count."""
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None")

    tokens = tokenizer.encode(text, lang=language)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

def add_padding(audio_segment, padding_ms):
    padding = AudioSegment.silent(duration=padding_ms)
    return audio_segment + padding

async def generate_audio_stream(text, language, speaker_wav_path, tokenizer=None):
    # ...
    audio_segments = [] # Lijst om audio segmenten op te slaan.
    paragraphs_and_sentences = split_text_into_paragraphs_and_sentences(text)
    for paragraph_index, sentences in enumerate(paragraphs_and_sentences):
        for sentence_index, sentence in enumerate(sentences):
            try:
                outputs = model.synthesize(
                    sentence,
                    config,
                    speaker_wav=speaker_wav_path,
                    language=language,
                )
                audio_data = outputs["wav"]
                audio_segments.append(audio_data) # Voeg audio segment toe.
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                yield b""

        if paragraph_index < len(paragraphs_and_sentences) - 1:
            paragraph_silence = AudioSegment.silent(duration=400)
            paragraph_silence_mp3_buffer = io.BytesIO()
            paragraph_silence.export(paragraph_silence_mp3_buffer, format="mp3", bitrate="320k", parameters=["-ar", "48000"])
            yield paragraph_silence_mp3_buffer.getvalue()

    # Overlap-adding:
    if audio_segments:
        combined_audio = overlap_add(audio_segments) # overlap add functie
        combined_audio = (combined_audio * 32767).astype(np.int16).tobytes()

        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)
                wav_file.writeframes(combined_audio)
            wav_buffer.seek(0)
            audio_segment = AudioSegment.from_wav(wav_buffer)
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3", bitrate="320k", parameters=["-ar", "48000"])
            yield mp3_buffer.getvalue()

def overlap_add(audio_segments, overlap_samples=200): # overlap samples aanpassen.
    combined_audio = np.concatenate(audio_segments) # combineer alle segmenten.
    return combined_audio

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

        audio_stream = generate_audio_stream(text, language, speaker_wav_path, tokenizer = tokenizer)
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
