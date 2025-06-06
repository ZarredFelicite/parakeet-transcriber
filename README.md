# Parakeet Audio Transcription

This project provides a tool for audio transcription using the NVIDIA NeMo Parakeet TDT (Text-Dependent Transcription) model. It can be used via a command-line interface or as a FastAPI web server.

## Features

*   Audio transcription using the powerful Parakeet TDT model.
*   Supports WAV and MP3 audio formats.
*   Automatic conversion to mono and resampling to 16kHz for compatibility with the ASR model.
*   Splits long audio files into segments for efficient processing.
*   Provides both a Command-Line Interface (CLI) and a FastAPI web server mode.
*   Basic error handling for file operations.
*   Cleans up temporary audio segment files after transcription.

## Prerequisites

This project uses Nix for environment management, which simplifies the setup process by handling dependencies, including CUDA, which is required for the NeMo model.

*   **Nix:** Ensure you have Nix installed on your system. Refer to the [official Nix installation guide](https://nixos.org/download/) for instructions.
*   **FFmpeg:** While the Nix shell should provide necessary libraries, ensure FFmpeg is available in your environment if you encounter issues with audio processing.

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

To transcribe a single audio file from the command line:

```bash
python parakeet.py [audio_filename.wav]
```

Replace `[audio_filename.wav]` with the path to your audio file (WAV or MP3).

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

The server exposes a `/transcribe` endpoint that accepts POST requests with an audio file in the `audio_file` form data field.

Example using `curl`:

```bash
curl -X POST -F "audio_file=@/path/to/your/audio.wav" http://127.0.0.1:5000/transcribe
```

The server will return a JSON response containing the transcription:

```json
{
  "transcription": "Your transcribed text here."
}
```

If an error occurs, the response will contain an "error" field.

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
