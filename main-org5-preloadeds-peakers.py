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

# Preload speakers
preloaded_speakers = {}

def preload_speakers():
    for filename in os.listdir(SPEAKER_DIR):
        if filename.endswith(".wav"):
            speaker_id = filename[:-4]
            speaker_wav_path = os.path.join(SPEAKER_DIR, filename)
            try:
                # Load the audio file (you can add more sophisticated preloading if needed)
                librosa.load(speaker_wav_path, sr=24000)  # Example: Load using librosa
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


async def generate_audio_stream(text, language, speaker_wav_path, tokenizer=None, chunk_size=32768):
    """
    Generates an audio stream from text, language, and speaker, yielding MP3 chunks in real-time.

    Changes:
    - Added chunk_size parameter to control the size of yielded MP3 chunks.
    - Modified the MP3 generation to stream chunks as they are created, instead of waiting for the entire file.
    """
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

    for paragraph_index, sentences in enumerate(paragraphs_and_sentences):
        tasks = [synthesize_sentence(sentence) for sentence in sentences]
        results = await asyncio.gather(*tasks)

        for audio_data in results:
            if audio_data is not None:
                audio_segments.append(audio_data)

        if paragraph_index < len(paragraphs_and_sentences) - 1:
            paragraph_silence = AudioSegment.silent(duration=400)
            mp3_silence_buffer = io.BytesIO()
            paragraph_silence.export(mp3_silence_buffer, format="mp3")
            yield mp3_silence_buffer.getvalue()

    if audio_segments:
        combined_audio = adaptive_overlap_add(audio_segments, min_overlap_samples=150, max_overlap_samples=300)
        combined_audio = (combined_audio * 32767).astype(np.int16)

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, combined_audio, 24000, format='wav')
            wav_buffer.seek(0)
            audio_segment = AudioSegment.from_wav(wav_buffer)
            mp3_buffer = io.BytesIO()

            # Stream chunks as they are produced
            for chunk in audio_segment.export(format="mp3", bitrate="128k", parameters=["-ar", "24000"]).read_chunks(chunk_size):
                yield chunk

def adaptive_overlap_add(audio_segments, min_overlap_samples=100, max_overlap_samples=400):
    if not audio_segments:
        return np.array([])

    # Calculate total length needed for output
    total_length = sum(len(segment) for segment in audio_segments)
    # Subtract potential overlaps
    total_length -= min_overlap_samples * (len(audio_segments) - 1)

    result = np.zeros(total_length)
    current_position = 0

    for i, segment in enumerate(audio_segments):
        if i == 0:
            # First segment goes in directly with no overlap
            result[:len(segment)] = segment
            current_position = len(segment)
        else:
            # For subsequent segments, create an overlap
            previous_segment_end_energy = np.mean(np.abs(audio_segments[i - 1][-max_overlap_samples:]))
            current_segment_start_energy = np.mean(np.abs(segment[:max_overlap_samples]))

            # Calculate adaptive overlap based on signal energy
            overlap_samples = int(min_overlap_samples + (max_overlap_samples - min_overlap_samples) *
                                   (1.0 - (previous_segment_end_energy + current_segment_start_energy) / 2))

            # Ensure overlap stays within bounds
            overlap_samples = min(max_overlap_samples, max(min_overlap_samples, overlap_samples))

            # Create crossfade window
            window = np.linspace(0, 1, overlap_samples)

            # Apply crossfade
            overlap_start = current_position - overlap_samples
            result[overlap_start:current_position] = (
                result[overlap_start:current_position] * (1 - window) +
                segment[:overlap_samples] * window
            )

            # Add the rest of the segment
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
            raise HTTPException(status_code=404, detail=f"Speaker ID '{speaker
