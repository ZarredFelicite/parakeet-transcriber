# Parakeet Audio Transcription

This project provides a tool for audio transcription using the NVIDIA NeMo Parakeet TDT (Text-Dependent Transcription) model. It can be used via a command-line interface or as a FastAPI web server.

## Features

*   Audio transcription using the powerful Parakeet TDT model.
*   Supports various audio (WAV, MP3) and video (MP4, MKV, AVI, MOV, WMV, FLV, WebM) formats by extracting the audio stream.
*   Automatic conversion to mono and resampling to 16kHz for compatibility with the ASR model.
*   Splits long audio files into segments for efficient processing.
*   Provides both a Command-Line Interface (CLI) and a FastAPI web server mode.
*   Basic error handling for file operations.
*   Cleans up temporary audio segment files after transcription.

## Prerequisites

This project uses Nix for environment management, which simplifies the setup process by handling dependencies, including CUDA, which is required for the NeMo model.

*   **Nix:** Ensure you have Nix installed on your system. Refer to the [official Nix installation guide](https://nixos.org/download/) for instructions.
*   **FFmpeg:** While the Nix shell should provide necessary libraries, ensure FFmpeg is available in your environment if you encounter issues with audio processing. The script now includes a check to verify that `ffmpeg` is in your system's PATH when the model is loaded.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  Enter the Nix development environment:
    ```bash
    nix develop
    ```
    This command will set up the required dependencies, including CUDA and Python with the necessary libraries. The `shellHook` in `flake.nix` will automatically create and activate a Python virtual environment using `uv` and install the project's dependencies from `pyproject.toml`.

## Usage

Once you are in the Nix development environment (`nix develop`), you can use the script in two modes:

### CLI Mode

To transcribe a single audio or video file from the command line:

```bash
python parakeet.py [input_file_path]
```

Replace `[input_file_path]` with the path to your audio (WAV, MP3) or video (MP4, MKV, AVI, MOV, WMV, FLV) file.

You can also specify the segment length for processing using the `--segment_length` argument (default is 60 seconds):

```bash
python parakeet.py my_audio.mp3 --segment_length 30
```

### Server Mode

To run the transcription as a FastAPI web server:

```bash
python parakeet.py --server
```

By default, the server will run on `http://0.0.0.0:5000`. You can customize the host and port using the `--host` and `--port` arguments:

```bash
python parakeet.py --server --host 127.0.0.1 --port 8000
```

The server exposes a `/transcribe` endpoint that accepts POST requests with an audio or video file in the `audio_file` form data field.

Example using `curl`:

```bash
curl -X POST -F "audio_file=@/path/to/your/input_file.mp4" http://127.0.0.1:5000/transcribe
```
Replace `.mp4` with the actual file extension of your audio or video file.

The server will return a JSON response containing the transcription:

```json
{
  "transcription": "Your transcribed text here."
}
```

If an error occurs, the response will contain an "error" field.

### Client Script for Streaming

A client script, `client.py`, is provided to demonstrate how to stream audio from a microphone to the WebSocket endpoint.

**Prerequisites:**

*   You will need to have `portaudio` installed on your system for `pyaudio` to work.
    *   On Debian/Ubuntu: `sudo apt-get install portaudio19-dev`
    *   On macOS (using Homebrew): `brew install portaudio`

**Usage:**

1.  Make sure the server is running (`python parakeet.py --server`).
2.  Run the client script:
    ```bash
    python client.py
    ```
    The script will start recording from your default microphone and stream the audio to the server. It will automatically stop after 5 seconds of silence.

### WebSocket Streaming Transcription V2 (Recommended)

A new WebSocket endpoint, `/ws/transcribe_v2`, is available for more robust and accurate real-time transcription.

**How it Works:**

1.  **Initial Padding:** The endpoint waits for 5 seconds of audio, padding with silence if necessary, to get a strong initial transcription.
2.  **Ever-Increasing Buffer:** After the initial transcription, it transcribes the entire audio buffer at regular intervals.
3.  **Stability Check:** A word is only sent to the client after it has appeared in the same position in the transcription for 3 consecutive rounds. This ensures that only "stable" words are sent, which significantly improves the quality of the real-time transcription.

To use this new endpoint, change the `WEBSOCKET_URI` in `client.py` to `ws://localhost:5000/ws/transcribe_v2`.

### WebSocket Streaming Transcription (Legacy)

The server also provides a WebSocket endpoint at `/ws/transcribe` for real-time streaming transcription. This is ideal for applications that need to process audio as it is being recorded or streamed.

**How it Works:**

1.  **Connection:** The client establishes a WebSocket connection to the server.
2.  **Audio Streaming:** The client sends audio chunks (16-bit, 16kHz, mono) to the server.
3.  **Sliding Window:** The server uses a 5-second sliding window to continuously process the incoming audio. The window slides by 1 second at a time.
4.  **Real-time Transcription:** The server transcribes the audio in the window and sends back only the newly transcribed words to the client. This minimizes data transfer and provides a smooth real-time experience.

**Note on `flake.nix` `start` script:** The `flake.nix` includes an application definition (`apps.parakeet`) that runs a `start` script. This script is configured to run the `parakeet.py` script in server mode on port 5001.

## Dependencies

The main Python dependencies are managed via `pyproject.toml` and installed by `uv` within the Nix development shell:

*   `numpy`
*   `torch`
*   `Cython`
*   `packaging`
*   `nemo_toolkit['asr']`
*   `fastapi`
*   `uvicorn`
*   `pydub`
*   `werkzeug`
*   `shutil` # For file handling in FastAPI endpoint

System-level dependencies, including CUDA and FFmpeg, are handled by the Nix environment defined in `flake.nix`.

## Error Handling

The script includes basic error handling for cases like the audio file not being found or issues during audio processing. Error messages will be printed to the console in CLI mode or returned in the JSON response in Server mode.

## Cleanup

Temporary audio segment files created during processing are automatically removed.
