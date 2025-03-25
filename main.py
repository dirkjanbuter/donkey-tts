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
import pylame

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

# Preload speakers
preloaded_speakers = {}

def preload_speakers():
    for filename in os.listdir(SPEAKER_DIR):
        if filename.endswith(".wav"):
            speaker_id = filename[:-4]
            speaker_wav_path = os.path.join(SPEAKER_DIR, filename)
            try:
                librosa.load(speaker_wav_path, sr=24000)
                preloaded_speakers[speaker_id] = speaker_wav_path
                print(f"Speaker '{speaker_id}' preloaded.")
            except Exception as e:
                print(f"Failed to preload speaker '{speaker_id}': {e}")

preload_speakers()

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

def convert_wav_to_mp3_lame(wav_data):
    """Converts WAV data to MP3 bytes using LAME."""
    try:
        data, samplerate = sf.read(io.BytesIO(wav_data))
        print(f"samplerate: {samplerate}, data shape: {data.shape}") #add print statement
        mp3_data = pylame.encode(data, samplerate, bitrate=192)
        return mp3_data
    except Exception as e:
        print(f"Error converting WAV to MP3: {e}, {traceback.format_exc()}") #add traceback info
        return None

async def generate_audio_stream(text, language, speaker_wav_path, tokenizer=None, chunk_size=32768):
    audio_segments = []
    paragraphs_and_sentences = split_text_into_paragraphs_and_sentences(text)

    async def synthesize_sentence(sentence):
        try:
            outputs = model.synthesize(
                sentence,
                config,
                speaker_wav=speaker_wav_path,
                language=language,
            )
            return outputs["wav"]
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            return None

    for sentences in paragraphs_and_sentences:
        tasks = [synthesize_sentence(sentence) for sentence in sentences]
        results = await asyncio.gather(*tasks)

        for audio_data in results:
            if audio_data is not None:
                audio_segments.append(audio_data)

        if audio_segments:
            combined_audio = adaptive_overlap_add(audio_segments, min_overlap_samples=150, max_overlap_samples=300)
            combined_audio = (combined_audio * 32767).astype(np.int16)

            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, combined_audio, 24000, format='wav')
                wav_buffer.seek(0)
                wav_data = wav_buffer.read()
                print(f"Wav data size: {len(wav_data)}") #Add print statement

                mp3_data = convert_wav_to_mp3_lame(wav_data)
                if mp3_data:
                    for i in range(0, len(mp3_data), chunk_size):
                        yield mp3_data[i:i + chunk_size]

            audio_segments = [] # reset audio_segments for next paragraph

def adaptive_overlap_add(audio_segments, min_overlap_samples=100, max_overlap_samples=400):
    if not audio_segments:
        return np.array([])

    total_length = sum(len(segment) for segment in audio_segments)
    total_length -= min_overlap_samples * (len(audio_segments) - 1)

    result = np.zeros(total_length)
    current_position = 0

    for i, segment in enumerate(audio_segments):
        if i == 0:
            result[:len(segment)] = segment
            current_position = len(segment)
        else:
            previous_segment_end_energy = np.mean(np.abs(audio_segments[i - 1][-max_overlap_samples:]))
            current_segment_start_energy = np.mean(np.abs(segment[:max_overlap_samples]))

            overlap_samples = int(min_overlap_samples + (max_overlap_samples - min_overlap_samples) *
                                    (1.0 - (previous_segment_end_energy + current_segment_start_energy) / 2))

            overlap_samples = min(max_overlap_samples, max(min_overlap_samples, overlap_samples))

            window = np.linspace(0, 1, overlap_samples)

            overlap_start = current_position - overlap_samples
            result[overlap_start:current_position] = (
                result[overlap_start:current_position] * (1 - window) +
                segment[:overlap_samples] * window
            )

            end_position = current_position - overlap_samples + len(segment)
            result[current_position:end_position] = segment[overlap_samples:]
            current_position = end_position

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
        speaker_wav_path = preloaded_speakers.get(speaker_id)
        if not speaker_wav_path:
            raise HTTPException(status_code=404, detail=f"Speaker ID '{speaker_id}' not found.")

        async def generate():
            yield b'\xFF\xFB\x90\x04\x00\x00\x00\x00\x00' # MP3 header
            async for chunk in generate_audio_stream(text, language, speaker_wav_path, tokenizer=tokenizer):
                yield chunk

        return Streaming
