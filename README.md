# Donkey TTS :: Real-Time Text-to-Speech Streaming with XTTS

This repository provides a FastAPI-based service for real-time text-to-speech (TTS) streaming using the XTTS model from Coqui TTS. It allows you to synthesize speech in various languages, cloning the voice of pre-loaded speakers from `.wav` files.

## Features

-   **Real-time Streaming:** Delivers audio as an MP3 stream, enabling immediate playback.
-   **Voice Cloning:** Uses XTTS to clone voices from provided speaker audio samples.
-   **Multi-language Support:** Synthesizes speech in multiple languages supported by XTTS.
-   **Paragraph and Sentence Handling:** Splits input text into paragraphs and sentences, generating audio with appropriate pauses.
-   **Speaker Management:** Loads speaker audio samples from a designated directory (`speakers/`) on application startup.
-   **Error Handling and Logging:** Provides robust error handling and logging for debugging and monitoring.
-   **FastAPI Integration:** Built with FastAPI for high performance and ease of use.

## Prerequisites

-   Python 3.8
-   CUDA-enabled GPU (recommended for performance)
-   PyTorch
-   Coqui TTS (`TTS`)
-   FastAPI
-   Pydantic
-   SoundFile
-   Pydub
-   Transformers

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/dirkjanbuter/donkey-tts.git
    cd donkey-tts
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install fastapi uvicorn pydantic TTS soundfile pydub transformers torch
    ```

4.  **Download the XTTS model:**

    Download the XTTS model from [Coqui TTS model releases](https://huggingface.co/coqui/XTTS-v2/tree/main) and place it in the `model/` directory.

5.  **Prepare speaker audio samples:**

    Place `.wav` files of speaker voice samples in the `speakers/` directory. The filenames (excluding the `.wav` extension) will be used as speaker IDs.

6.  Optional: Docker

    docker compose build<br>
    docker compose up

## Usage

1.  **Run the FastAPI application:**

    ```bash
    uvicorn main:app --reload
    ```

    (Replace `main` with the name of your python file if it is different)

2.  **Send a POST request to the `/tts_stream/` endpoint:**

    Use a tool like `curl` or Postman to send a POST request with the following form data:

    -   `text`: The text to be synthesized.
    -   `language`: The language of the text.
    -   `speaker_id`: The ID of the speaker (filename without `.wav`).

    Example `curl` command:

    ```bash
    curl -X POST -F "text=Hello, this is a test." -F "language=en" -F "speaker_id=yvonta" http://127.0.0.1:8979/tts_stream/ > output.mp3
    ```

    This will save the generated audio stream to `output.mp3`.

    ```bash
    curl --connect-timeout 30 --max-time 0 -X POST -F "text=Welcome to Donkey TTS!" -F "language=en" -F "speaker_id=yvonta" http://127.0.0.1:8979/tts_stream/ | mpg123 -q -
    ```

    This wil stream and play the text in realtime


## Speaker Management

-      Place your speaker's `.wav` files inside the `speakers/` folder.
-      The application loads these speakers into memory on startup.
-      The filename (without the `.wav` extension) becomes the speaker ID.

## Notes

-      Ensure you have a compatible GPU and CUDA setup for optimal performance.
-      The XTTS model requires significant GPU memory.
-      Adjust padding and silence durations in the code as needed for your specific use case.
-      Ensure that your speaker files are 24000hz, mono, and in wav format.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug fixes, feature requests, or improvements.
