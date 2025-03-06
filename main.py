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
import subprocess

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

async def generate_audio_stream(text, language, speaker_wav_path, tokenizer=None):
    audio_segments = []
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
                audio_segments.append(audio_data)
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                yield b""

        if paragraph_index < len(paragraphs_and_sentences) - 1:
            paragraph_silence = AudioSegment.silent(duration=400)
            opus_silence_buffer = io.BytesIO()
            paragraph_silence.export(opus_silence_buffer, format="ogg", codec="libopus", parameters=["-ar", "24000"])
            yield opus_silence_buffer.getvalue()

    if audio_segments:
        # Vervang overlap_add door adaptive_overlap_add
        combined_audio = adaptive_overlap_add(audio_segments, min_overlap_samples=150, max_overlap_samples=300)
        combined_audio = (combined_audio * 32767).astype(np.int16)

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, combined_audio, 24000, format='wav')
            wav_buffer.seek(0)
            audio_segment = AudioSegment.from_wav(wav_buffer)
            opus_buffer = io.BytesIO()
            audio_segment.export(opus_buffer, format="ogg", codec="libopus", parameters=["-ar", "24000"])
            yield opus_buffer.getvalue()

def overlap_add(audio_segments, overlap_samples=200):
    combined_audio = np.concatenate(audio_segments)
    result = np.zeros_like(combined_audio)
    hop_length = len(audio_segments[0]) - overlap_samples
    window = librosa.filters.get_window("hann", overlap_samples * 2)

    for i, segment in enumerate(audio_segments):
        start = i * hop_length
        end = start + len(segment)

        if i > 0:
            overlap_start = start
            overlap_end = overlap_start + overlap_samples
            result[overlap_start:overlap_end] += segment[:overlap_samples] * window[:overlap_samples]

        result[start + (overlap_samples if i > 0 else 0):end] += segment[overlap_samples if i > 0 else 0:]

    return result

def adaptive_overlap_add(audio_segments, min_overlap_samples=100, max_overlap_samples=400):
    combined_audio = np.concatenate(audio_segments)
    result = np.zeros_like(combined_audio)
    hop_length = len(audio_segments[0]) - max_overlap_samples

    for i, segment in enumerate(audio_segments):
        start = i * hop_length
        end = start + len(segment)

        if i > 0:
            # Bereken de energie aan het einde van het vorige segment
            previous_segment_end_energy = np.mean(np.abs(audio_segments[i - 1][-max_overlap_samples:]))

            # Bereken de energie aan het begin van het huidige segment
            current_segment_start_energy = np.mean(np.abs(segment[:max_overlap_samples]))

            # Bereken de optimale overlap op basis van de energie
            overlap_samples = int(min_overlap_samples + (max_overlap_samples - min_overlap_samples) * (previous_segment_end_energy + current_segment_start_energy) / 2)

            # Zorg ervoor dat de overlap binnen de grenzen blijft
            overlap_samples = min(max_overlap_samples, max(min_overlap_samples, overlap_samples))

            overlap_start = start
            overlap_end = overlap_start + overlap_samples
            window = librosa.filters.get_window("hann", overlap_samples * 2)
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
        return StreamingResponse(audio_stream, media_type="audio/ogg")

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
