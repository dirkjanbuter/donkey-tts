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
from pydub import AudioSegment

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

def filteraudio(frame, framecount, numframes, max_values):
    """
    Python translation of the C code for audio volume normalization, with unused variables removed.

    Args:
        frame: A list or numpy array of floats representing the audio frame.
        framecount: An integer representing the frame count.
        numframes: An integer representing the number of frames to consider for maximum.
        max_values: a list to store the maximum values. It should be initialized outside the function.

    Returns:
        True (or any non-zero value, as per the original C code).
    """

    max_values[framecount % numframes] = 0.0
    for i in range(2048):
        val = abs(frame[i])
        if val > max_values[framecount % numframes]:
            max_values[framecount % numframes] = val

    if framecount <= numframes:
        maxmax = 1.0
    else:
        maxmax = max(max_values[i] for i in range(min(numframes, framecount)))

    if maxmax < 0.01:
        maxmax = 0.2
    if maxmax > 0.1:
        maxmax = 0.1

    for i in range(2048):
        frame[i] *= 1.0 / maxmax

    return True
    
def amplify_audio(audio_data, numframes=10):
    """
    Amplifies audio data using the filteraudio function.

    Args:
        audio_data: A numpy array of floats representing the audio data.
        numframes: The number of frames to consider for maximum.

    Returns:
        A numpy array of floats representing the amplified audio data.
    """

    frame_size = 2048
    num_frames = len(audio_data) // frame_size
    max_values = [0.0] * numframes
    amplified_data = np.copy(audio_data) #create a copy to prevent in place modification.

    for framecount in range(num_frames):
        frame = amplified_data[framecount * frame_size: (framecount + 1) * frame_size]
        filteraudio(frame, framecount, numframes, max_values)
        amplified_data[framecount * frame_size: (framecount + 1) * frame_size] = frame #assign the modified frame back.
    return amplified_data[:num_frames * frame_size] #return only the frames that were processed.    

def convert_wav_to_mp3_pymp3(wav_data):
    """Converts WAV data to MP3 bytes using pydub."""
    try:
        data, samplerate = sf.read(io.BytesIO(wav_data))
        print(f"samplerate: {samplerate}, data shape: {data.shape}")

        # Convert numpy array to pydub AudioSegment
        audio_segment = AudioSegment(
            data.tobytes(),
            frame_rate=samplerate,
            sample_width=data.dtype.itemsize,
            channels=1 if len(data.shape) == 1 else data.shape[1]
        )

        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3", bitrate="192k")
        mp3_data = mp3_buffer.getvalue()
        return mp3_data

    except Exception as e:
        print(f"Error converting WAV to MP3: {e}, {traceback.format_exc()}")
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
            
            # Amplify the audio here
            combined_audio = amplify_audio(combined_audio)
            
            combined_audio = (combined_audio * 32767).astype(np.int16)

            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, combined_audio, 24000, format='wav')
                wav_buffer.seek(0)
                wav_data = wav_buffer.read()
                print(f"Wav data size: {len(wav_data)}") #Add print statement

                mp3_data = convert_wav_to_mp3_pymp3(wav_data)
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

        return StreamingResponse(generate(), media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"Error processing TTS stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS stream generation failed: {str(e)}")
