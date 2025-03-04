from fastapi import FastAPI, HTTPException, Response, Request, Form
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
import numpy as np
import logging
import re
import asyncio
from fastapi.responses import StreamingResponse
import wave
from transformers import AutoTokenizer

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

# Dictionary to store speaker audio data
speaker_audio_data = None  # Initialize to None

# Load speakers on application startup
@app.on_event("startup")
async def startup_event():
    global speaker_audio_data  # Declare as global to modify
    speaker_audio_data = {}  # Initialize the dictionary
    try:
        for filename in os.listdir(SPEAKER_DIR):
            if filename.endswith(".wav"):
                speaker_id = filename[:-4]
                speaker_path = os.path.join(SPEAKER_DIR, filename)
                audio_data, samplerate = sf.read(speaker_path)
                speaker_audio_data[speaker_id] = (audio_data, samplerate)
        print("Speakers loaded into memory.")
    except Exception as e:
        logger.error(f"Error loading speakers: {e}", exc_info=True)

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

def add_padding(audio_segment, padding_ms):
    padding = AudioSegment.silent(duration=padding_ms)
    return audio_segment + padding

async def generate_audio_stream(text, language, speaker_id, tokenizer=None):
    try:
        if speaker_id not in speaker_audio_data:
            raise HTTPException(status_code=404, detail=f"Speaker ID '{speaker_id}' not found")

        speaker_audio, samplerate = speaker_audio_data[speaker_id]

        paragraphs_and_sentences = split_text_into_paragraphs_and_sentences(text)
        for paragraph_index, sentences in enumerate(paragraphs_and_sentences):
            for sentence_index, sentence in enumerate(sentences):
                try:
                    outputs = model.synthesize(
                        sentence,
                        config,
                        speaker_wav=speaker_audio,
                        language=language,
                        speaker_sample_rate=samplerate
                    )
                    audio_data = outputs["wav"]
                    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()

                    with io.BytesIO() as wav_buffer:
                        with wave.open(wav_buffer, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(24000)
                            wav_file.writeframes(audio_bytes)
                        wav_audio = AudioSegment.from_wav(wav_buffer)
                        wav_audio = add_padding(wav_audio, 200)
                        mp3_buffer = io.BytesIO()
                        wav_audio.export(mp3_buffer, format="mp3", bitrate="128k", parameters=["-ar", "24000"])
                        yield mp3_buffer.getvalue()

                    if sentence_index < len(sentences) - 1:
                        silence = AudioSegment.silent(duration=0)
                        silence_mp3_buffer = io.BytesIO()
                        silence.export(silence_mp3_buffer, format="mp3", bitrate="128k", parameters=["-ar", "24000"])
                        yield silence_mp3_buffer.getvalue()

                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    yield b""

            if paragraph_index < len(paragraphs_and_sentences) - 1:
                paragraph_silence = AudioSegment.silent(duration=400)
                paragraph_silence_mp3_buffer = io.BytesIO()
                paragraph_silence.export(paragraph_silence_mp3_buffer, format="mp3", bitrate="128k", parameters=["-ar", "24000"])
                yield paragraph_silence_mp3_buffer.getvalue()

    except Exception as e:
        logger.error(f"Error in generate_audio_stream: {e}", exc_info=True)
        yield b""

@app.post("/tts_stream/")
async def text_to_speech_stream(
    text: str = Form(...),
    language: str = Form(...),
    speaker_id: str = Form(...),
    req: Request = None
):
    try:
        print(f"Received TTS streaming request for speaker: {speaker_id}, language: {language}")
        audio_stream = generate_audio_stream(text, language, speaker_id, tokenizer=tokenizer)
        return StreamingResponse(audio_stream, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"Error in TTS streaming processing: {e}", exc_info=True)
        return Response(
            content=json.dumps({"detail": str(e)}),
            media_type="application/json",
            status_code=500,
        )
