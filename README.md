# Parakeet Audio Transcription

This project provides a comprehensive audio transcription solution using the NVIDIA NeMo Parakeet TDT (Text-Dependent Transcription) model. It supports both command-line usage and a FastAPI web server with multiple endpoints for different transcription needs.

## Features

### Core Transcription
* Audio transcription using the powerful Parakeet TDT 0.6B v2 model
* Supports various audio (WAV, MP3) and video (MP4, MKV, AVI, MOV, WMV, FLV, WebM) formats by extracting the audio stream
* Automatic conversion to mono and resampling to 16kHz for compatibility with the ASR model
* Splits long audio files into segments for efficient processing (CLI and POST endpoint)

### Server Modes
* **Command-Line Interface (CLI)** for single file transcription
* **FastAPI web server** with multiple endpoints:
  * `/transcribe` - Basic batch transcription of uploaded files
  * `/transcribe_timestamps` - Transcription with segment-level timestamps
  * `/transcribe_diarize` - Transcription with speaker diarization using pyannote.audio
  * `/transcribe_cluster` - Transcription with speaker clustering using embeddings (ECAPA/TitaNet)
  * `/ws/transcribe` - Real-time streaming transcription via WebSocket

### Advanced Features
* **Real-time transcription** with advanced word stabilization and context-aware punctuation
* **Speaker diarization** using pyannote.audio pipeline for identifying different speakers
* **Speaker clustering** using ECAPA or TitaNet embeddings with HDBSCAN clustering
* **OpenAI integration** for inferring speaker names from conversation context
* **Embedding visualization** with t-SNE/PCA projection plots
* **Streaming client** with voice activity detection and typing mode

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

The server exposes multiple endpoints:

#### 1. POST `/transcribe`

Basic batch transcription endpoint that accepts audio or video files.

```bash
curl -X POST -F "audio_file=@/path/to/your/input_file.mp4" http://127.0.0.1:5000/transcribe
```

Response:
```json
{
  "transcription": "Your transcribed text here."
}
```

#### 2. POST `/transcribe_timestamps`

Transcription with segment-level timestamps from the NeMo model.

```bash
curl -X POST -F "audio_file=@/path/to/your/input_file.mp4" http://127.0.0.1:5000/transcribe_timestamps
```

Response:
```json
{
  "transcription": "Full transcription text",
  "segments": [
    {"start": 0.0, "end": 4.23, "text": "first spoken segment"},
    {"start": 4.23, "end": 8.45, "text": "second spoken segment"}
  ]
}
```

#### 3. POST `/transcribe_diarize`

Transcription with speaker diarization using pyannote.audio pipeline. Requires `HUGGINGFACE_ACCESS_TOKEN` in environment or `.env` file.

```bash
curl -X POST -F "audio_file=@/path/to/your/input_file.mp4" http://127.0.0.1:5000/transcribe_diarize
```

Response:
```json
{
  "segments": [
    {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00", "text": "Hello there.", "segments": [...]},
    {"start": 3.2, "end": 7.5, "speaker": "SPEAKER_01", "text": "Hi!", "segments": [...]}
  ]
}
```

#### 4. POST `/transcribe_cluster`

Transcription with speaker clustering using embeddings (ECAPA or TitaNet models).

```bash
curl -X POST -F "audio_file=@/path/to/your/input_file.mp4" \
  "http://127.0.0.1:5000/transcribe_cluster?model=ecapa&name_labels=true"
```

Query parameters:
- `model`: `ecapa`, `titanet`, or `combo` (default: `ecapa`)
- `name_labels`: `true` to use OpenAI for name inference (requires `OPENAI_API_KEY`)

#### 5. WebSocket `/ws/transcribe`

Real-time streaming transcription endpoint with advanced word stabilization.

**Connection:** `ws://<host>:<port>/ws/transcribe?chunk_duration_ms=250`

The client streams 16-bit, 16kHz, mono PCM audio chunks. The server applies word stabilization and contextual punctuation, sending back confirmed words in real-time.

### Client Script for Streaming (`client.py`)

A sophisticated client script for real-time audio streaming with voice activity detection and multiple output modes.

**Features:**
- Voice Activity Detection (VAD) with configurable aggressiveness
- Automatic silence detection and stopping (5 seconds of silence)
- Typing mode that inputs transcription directly into focused applications
- Verbose mode with timing and profiling information
- Configurable chunk duration and server connection

**Usage:**

1. Start the server:
   ```bash
   nix run .#parakeet  # or python parakeet.py --server
   ```

2. Run the client:
   ```bash
   python client.py [options]
   ```

**Options:**
- `--send_interval_ms 250`: Audio chunk duration in milliseconds
- `--verbose`: Enable detailed logging and timing information
- `--type [delay]`: Type transcription into focused window (requires `wtype`)
- `--host localhost`: Server hostname
- `--port 5000`: Server port

**Examples:**
```bash
# Basic usage - prints transcription to stdout
python client.py

# Verbose mode with timing information
python client.py --verbose

# Type transcription into focused application with 10ms delay
python client.py --type 10

# Connect to remote server
python client.py --host 192.168.1.100 --port 5001
```

**Typing Mode:**
When using `--type`, the client requires `wtype` (Wayland) for typing functionality. The transcription will be typed directly into the currently focused application instead of printed to stdout.

## Project Structure

```
asr2/
├── parakeet.py          # Main server with CLI and FastAPI endpoints
├── client.py            # Streaming client with VAD and typing support
├── flake.nix           # Nix development environment and dependencies
├── pyproject.toml      # Python project metadata and dependencies
├── uv.lock            # Locked dependency versions
├── .env               # Environment variables (HuggingFace, OpenAI tokens)
├── .gitignore         # Git ignore patterns
├── LICENSE            # Project license
└── README.md          # This file
```

### Key Components

**`parakeet.py`** - Main application with:
- CLI mode for single file transcription
- FastAPI server with 5 different endpoints
- Advanced WebSocket streaming with word stabilization
- Speaker diarization and clustering capabilities
- OpenAI integration for speaker name inference

**`client.py`** - Streaming client featuring:
- Voice Activity Detection (WebRTC VAD)
- Automatic silence detection and stopping
- Real-time audio streaming to WebSocket
- Typing mode for direct input to applications
- Comprehensive timing and profiling options

## Configuration

### Environment Variables

Create a `.env` file in the project root for optional features:

```bash
# Required for speaker diarization (/transcribe_diarize)
HUGGINGFACE_ACCESS_TOKEN=your_hf_token_here

# Required for OpenAI name inference (name_labels=true in /transcribe_cluster)
OPENAI_API_KEY=your_openai_api_key_here
```

### Dependencies

**Python Dependencies** (managed via `pyproject.toml` and `uv`):
- Core: `numpy`, `torch`, `nemo_toolkit[asr]`
- Web: `fastapi`, `uvicorn`, `websockets`, `python-multipart`
- Audio: `pyaudio`, `webrtcvad`, `pydub`
- ML/AI: `pyannote.audio`, `hdbscan`, `scikit-learn`, `matplotlib`
- Utils: `Cython`, `packaging`, `dotenv`, `evdev-binary`

**System Dependencies** (handled by Nix):
- CUDA toolkit and runtime for GPU acceleration
- FFmpeg for audio/video processing
- PortAudio for microphone input
- Various system libraries (zlib, libgcc, X11, etc.)

## Advanced Usage

### Speaker Diarization vs Clustering

**Diarization (`/transcribe_diarize`):**
- Uses pyannote.audio's pre-trained pipeline
- More accurate for well-separated speakers
- Requires HuggingFace token
- Slower processing time

**Clustering (`/transcribe_cluster`):**
- Uses speaker embeddings (ECAPA/TitaNet) with HDBSCAN
- Better for overlapping speech or noisy audio
- Faster processing
- Optional OpenAI name inference

### Embedding Models

- **ECAPA**: Fast, good general performance
- **TitaNet**: Higher quality, slower processing
- **Combo**: Concatenated ECAPA + TitaNet embeddings

### Performance Tuning

- Adjust `--segment_length` for memory usage (default: 60 seconds)
- Use `--verbose` for detailed timing information
- GPU acceleration is automatic when CUDA is available
- WebSocket chunk duration affects latency vs accuracy trade-off

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--segment_length` parameter
2. **FFmpeg not found**: Ensure you're in the Nix development shell
3. **HuggingFace token error**: Set `HUGGINGFACE_ACCESS_TOKEN` in `.env`
4. **Audio device issues**: Check microphone permissions and availability
5. **Typing mode not working**: Install `wtype` for Wayland or check X11 setup

### Error Handling

- CLI mode: Errors printed to console with detailed messages
- Server mode: HTTP error responses with status codes and details
- WebSocket: Connection errors handled gracefully with reconnection support
- Temporary files: Automatically cleaned up after processing
