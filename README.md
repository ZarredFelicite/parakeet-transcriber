# Parakeet Audio Transcription

This project provides a tool for audio transcription using the NVIDIA NeMo Parakeet TDT (Text-Dependent Transcription) model. It can be used via a command-line interface or as a FastAPI web server with a WebSocket endpoint for real-time streaming.

## Features

* Audio transcription using the powerful Parakeet TDT model.
* Supports various audio (WAV, MP3) and video (MP4, MKV, AVI, MOV, WMV, FLV, WebM) formats by extracting the audio stream.
* Automatic conversion to mono and resampling to 16kHz for compatibility with the ASR model.
* Splits long audio files into segments for efficient processing (CLI and POST endpoint).
* Provides both a Command-Line Interface (CLI) and a FastAPI web server mode.
* FastAPI server includes:
  * A POST endpoint (`/transcribe`) for batch transcription of uploaded files.
  * A WebSocket endpoint (`/ws/transcribe`) for real-time streaming transcription.
* Real-time transcription features advanced word stabilization and context-aware punctuation.

## Getting Started

### Prerequisites

This project uses Nix for environment management. This simplifies the setup process by handling all dependencies, including CUDA, which is required for the NeMo model.

* **Nix:** Ensure you have Nix installed on your system. Refer to the [official Nix installation guide](https://nixos.org/download/) for instructions.
* **FFmpeg:** The Nix shell provides the necessary libraries, but if you encounter audio processing issues, ensure `ffmpeg` is available in your environment. The script includes a check to verify that `ffmpeg` is in your system's PATH.

### Installation & Setup

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Enter the Nix development environment:

   ```bash
   nix develop
   ```

   This command sets up all required dependencies, including CUDA and Python. A Python virtual environment will be automatically created and activated using `uv`, and the project's dependencies from `pyproject.toml` will be installed.

## Usage

Once you are in the Nix development environment (`nix develop`), you can use the script in two modes:

### CLI Mode

To transcribe a single audio or video file from the command line:

```bash
python parakeet.py [input_file_path]
```

Replace `[input_file_path]` with the path to your audio (WAV, MP3) or video (MP4, MKV, AVI, MOV, WMV, FLV) file.

You can also specify the segment length for processing using the `--segment_length` argument (default is 60 seconds), and enable verbose logging with `--verbose`:

```bash
python parakeet.py my_audio.mp3 --segment_length 30 --verbose
```

### Server Mode

To run the transcription as a FastAPI web server, you can use the `start` script defined in `flake.nix`, which is the recommended method:

```bash
nix run .#parakeet
```

This will start the server on `http://0.0.0.0:5001` by default.

Alternatively, you can run the script directly:

```bash
python parakeet.py --server
```

By default, the server will run on `http://0.0.0.0:5000`. You can customize the host, port, segment length for POST requests, and verbosity:

```bash
python parakeet.py --server --host 127.0.0.1 --port 8000 --segment_length 45 --verbose
```

The server exposes two main endpoints:

#### 1. POST `/transcribe`

Accepts POST requests with an audio or video file in the `audio_file` form data field for batch transcription.

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

#### 2. WebSocket `/ws/transcribe`

Provides a WebSocket endpoint for real-time streaming transcription. This is ideal for applications that need to process audio as it is being recorded or streamed.

**How it Works:**

The client establishes a WebSocket connection to `ws://<host>:<port>/ws/transcribe`. It can pass `chunk_duration_ms` and `verbose` as query parameters. The client then streams audio chunks (16-bit, 16kHz, mono PCM). The server transcribes the audio, applying word stabilization and contextual punctuation, and sends back confirmed words in real-time.

### Client Script for Streaming (`client.py`)

A client script, `client.py`, is provided to demonstrate how to stream audio from a microphone to the WebSocket endpoint (`/ws/transcribe`).

**Prerequisites:**

The Nix environment provides all necessary dependencies. However, if you run the client outside of the managed environment, you may need to install `portaudio`:

* On Debian/Ubuntu: `sudo apt-get install portaudio19-dev`
* On macOS (using Homebrew): `brew install portaudio`

**Usage:**

1. Make sure the server is running (`nix run .#parakeet` or `python parakeet.py --server`).

2. Run the client script:

   ```bash
   python client.py
   ```

   The script will start recording from your default microphone and stream the audio to the server. It will print the transcribed words as they are received. The client automatically stops after 5 seconds of silence. The WebSocket URI is configurable and defaults to `ws://localhost:5000/ws/transcribe?chunk_duration_ms=250`.

## Project Structure

* `parakeet.py`: The main script containing the CLI and FastAPI server logic.
* `client.py`: An example client for the real-time streaming WebSocket endpoint.
* `flake.nix`: The Nix flake for managing the development environment and dependencies.
* `pyproject.toml`: Defines the Python project metadata and dependencies.
* `README.md`: You are here!

## Dependencies

The main Python dependencies are managed via `pyproject.toml` and installed by `uv` within the Nix development shell:

* `numpy`
* `torch`
* `Cython`
* `packaging`
* `nemo_toolkit['asr']`
* `fastapi`
* `uvicorn[standard]` (for WebSocket support)
* `pydub`
* `python-multipart` (for FastAPI file uploads)
* `websockets` (though `uvicorn[standard]` should cover this for server-side)
* `pyaudio` (for `client.py`)

System-level dependencies, including CUDA and FFmpeg, are handled by the Nix environment defined in `flake.nix`.

## Error Handling

The script includes basic error handling for cases like the audio file not being found or issues during audio processing. Error messages will be printed to the console in CLI mode or returned in the JSON response in Server mode.

## Cleanup

Temporary audio segment files created during processing are automatically removed.
