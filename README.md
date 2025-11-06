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
* **Real-time transcription** with advanced word stabilization using frequency-based confirmation system
* **Multi-round tracking** with consecutive observation requirements to prevent intermittent word graduation
* **Position consistency tracking** to ensure words appear at stable positions across transcription rounds
* **Automatic alignment recovery** with partial-match fallback and stall detection
* **Flexible number matching** for handling variations like "$3" vs "$3.60" or "three dollars" vs "$3"
* **Context-aware word tracking** (experimental) for distinguishing identical words in different contexts
* **Speaker diarization** using pyannote.audio pipeline for identifying different speakers
* **Speaker clustering** using ECAPA or TitaNet embeddings with HDBSCAN clustering
* **k-NN noise filling** to assign cluster labels to HDBSCAN outlier points
* **OpenAI integration** for inferring speaker names from conversation context using GPT
* **Embedding visualization** with automatic t-SNE/PCA projection plots for cluster analysis
* **Streaming client** with voice activity detection (WebRTC VAD), automatic silence detection, and typing mode
* **Stream vs Batch testing** mode to validate streaming transcription accuracy against batch results

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

**Stream Testing Mode:**

To validate the streaming transcription system by comparing it against batch transcription:

```bash
python parakeet.py my_audio.mp3 --stream-test --verbose
```

This mode:
- Runs a batch transcription first to get the reference output
- Simulates streaming transcription by processing the audio in chunks
- Compares outputs in real-time and stops at the first divergence
- Displays detailed diagnostics including word states, alignment points, and context
- Uses color-coded visualization: 🟢 confirmed words, 🟡 potential words, 🔴 new words, 🔵 alignment points

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

Response includes an optional `embedding_plot` field with path to a saved visualization PNG showing the 2-D projection of speaker embeddings (uses t-SNE for ≤100 points, PCA for larger datasets).

#### 5. WebSocket `/ws/transcribe`

Real-time streaming transcription endpoint with advanced word stabilization and confirmation system.

**Connection:** `ws://<host>:<port>/ws/transcribe?chunk_duration_ms=250`

Query parameters:
- `chunk_duration_ms`: Duration of audio chunks sent by client (default: 250ms)

The client streams 16-bit, 16kHz, mono PCM audio chunks. The server applies:
- **Word frequency tracking**: Words must appear multiple times (min 4 by default) to be confirmed
- **Consecutive round observation**: Words must appear in consecutive transcription rounds (min 2 by default)
- **Position consistency**: Words must stabilize at approximately the same position in the transcript
- **Automatic alignment**: Recovers from ASR output variations using prefix matching and partial sequence alignment
- **Stall detection**: Forces progress if alignment gets stuck at the same position for too many rounds (10+ rounds)
- **Flexible matching**: Handles number variations and punctuation differences

The server sends back only fully confirmed words, preventing spurious or unstable words from appearing in the output.

### Client Script for Streaming (`client.py`)

A sophisticated client script for real-time audio streaming with voice activity detection and multiple output modes.

**Features:**
- Voice Activity Detection (VAD) with WebRTC VAD, configurable aggressiveness (0-3, default: 3)
- Automatic silence detection and stopping (5 seconds of continuous silence)
- Typing mode that inputs transcription directly into focused applications using `wtype` (Wayland)
- Character-by-character output with timing (10ms per character, 50ms per space)
- Verbose mode with detailed timing and profiling information (startup time, WebSocket connection, etc.)
- Configurable audio chunk duration (default: 250ms)
- Configurable server host and port for remote connections
- ALSA/JACK error suppression (shown only in verbose mode)

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
├── parakeet.py                      # Main server with CLI and FastAPI endpoints
├── parakeet_backup.py              # Backup of previous version
├── client.py                        # Streaming client with VAD and typing support
├── context_aware_tracker.py        # Experimental context-aware word tracking
├── flake.nix                       # Nix development environment and dependencies
├── flake.lock                      # Nix flake lock file
├── pyproject.toml                  # Python project metadata and dependencies
├── .env                            # Environment variables (HuggingFace, OpenAI tokens)
├── .gitignore                      # Git ignore patterns
├── LICENSE                         # Project license
├── README.md                       # This file
│
├── Debug & Test Scripts:
├── debug_alignment.py              # Debug alignment logic
├── debug_context_divergence.py     # Debug context-aware tracking
├── debug_graduation_blocking.py    # Debug word graduation issues
├── debug_no_graduation.py          # Debug why words aren't graduating
├── debug_streaming_chunks.py       # Debug chunk processing
├── debug_streaming_params.py       # Debug streaming parameters
├── debug_youre_repetition.py       # Debug repeated word handling
├── test_context_divergence.py      # Test context-aware divergence
├── test_context_integration.py     # Test context-aware integration
├── test_disable_context.py         # Test with context tracking disabled
├── test_final_context.py           # Final context tracking tests
├── test_integrated_context.py      # Integrated context tracking tests
├── test_position_tolerance.py      # Test position consistency tolerance
├── test_repeated_words.py          # Test repeated word handling
├── test_stall_detection.py         # Test stall detection mechanism
└── tmp_test_tracker.py             # Temporary tracker testing
```

**Note:** Debug and test scripts are development tools not required for normal operation.

### Key Components

**`parakeet.py`** - Main application with:
- CLI mode for single file transcription
- FastAPI server with 5 different endpoints
- Advanced WebSocket streaming with multi-layer word stabilization:
  - `TranscriptionTracker` class with frequency, consecutive round, and position tracking
  - `WordState` class tracking individual word instances across rounds
  - Automatic alignment with prefix matching and partial sequence recovery
  - Stall detection and forced progress mechanism
  - Flexible number and punctuation matching
- Speaker diarization using pyannote.audio pipeline
- Speaker clustering using ECAPA/TitaNet embeddings with HDBSCAN
- k-NN noise point classification for clustering outliers
- OpenAI integration for speaker name inference using GPT
- Embedding visualization with automatic plot generation
- Stream testing mode for validation and debugging
- Progress bar suppression for clean terminal output

**`client.py`** - Streaming client featuring:
- Voice Activity Detection using WebRTC VAD (aggressiveness level 3)
- Automatic silence detection (5-second timeout)
- Real-time audio streaming to WebSocket (configurable chunk size)
- Typing mode for direct input to Wayland applications via `wtype`
- Character-by-character output with natural timing
- Comprehensive timing and profiling (startup, connection, transcription)
- Configurable remote server connection (host/port)
- ALSA/JACK error suppression

## Configuration

### Environment Variables

Create a `.env` file in the project root for optional features:

```bash
# Required for speaker diarization (/transcribe_diarize)
HUGGINGFACE_ACCESS_TOKEN=your_hf_token_here

# Required for OpenAI name inference (name_labels=true in /transcribe_cluster)
OPENAI_API_KEY=your_openai_api_key_here
```

The application uses `python-dotenv` to automatically load these variables from `.env` when needed.

### Transcription Tracker Configuration

The `TranscriptionTracker` class can be configured with these parameters:

```python
tracker = TranscriptionTracker(
    min_confirmed_words=4,        # Minimum confirmed words before allowing removal
    min_frequency=4,              # Minimum observations before word can graduate
    min_consecutive_rounds=2      # Consecutive rounds word must appear
)
```

These are set as defaults in the WebSocket endpoint and can be modified in `parakeet.py`.

### Dependencies

**Python Dependencies** (managed via `pyproject.toml` and `uv`):
- Core: `numpy<2.0`, `torch`, `nemo_toolkit[asr]`
- Web: `fastapi`, `uvicorn`, `websockets`, `python-multipart`
- Audio: `pyaudio`, `webrtcvad`, `pydub`
- ML/AI: `pyannote.audio`, `hdbscan`, `scikit-learn`, `matplotlib`
- Utils: `Cython`, `packaging`, `dotenv`, `evdev-binary`

**Note:** NumPy is pinned to `<2.0` for compatibility with NeMo toolkit.

**System Dependencies** (handled by Nix):
- CUDA toolkit and runtime for GPU acceleration
- FFmpeg for audio/video processing
- PortAudio for microphone input
- Various system libraries (zlib, libgcc, X11, etc.)

## Technical Details

### Model Information

**ASR Model:** NVIDIA NeMo Parakeet TDT 0.6B v2
- Parameters: 600M
- Architecture: Transformer-based encoder-decoder
- Sample rate: 16kHz mono
- Vocabulary: Character-based with punctuation
- Automatically moved to CUDA if available

**Diarization Pipeline:** pyannote/speaker-diarization-3.1
- Requires HuggingFace authentication token
- Automatic device detection (CUDA/CPU)

**Speaker Embedding Models:**
- ECAPA: speechbrain/spkrec-ecapa-voxceleb
- TitaNet: nvidia/titanet_large (NeMo)

### Audio Processing

All audio is automatically normalized to:
- **Sample rate:** 16kHz
- **Channels:** Mono (stereo downmixed)
- **Format:** 16-bit PCM WAV
- **Codec:** pcm_s16le

Video files are processed by:
1. Extracting audio stream with FFmpeg
2. Converting to 16kHz mono WAV
3. Processing through ASR model
4. Cleaning up temporary files

### WebSocket Protocol

The streaming endpoint expects:
- **Format:** Raw PCM audio bytes
- **Encoding:** 16-bit signed little-endian
- **Sample rate:** 16kHz
- **Channels:** 1 (mono)
- **Chunk size:** Configurable via `chunk_duration_ms` parameter

Returns space-separated confirmed words as text messages.

## Advanced Usage

### Speaker Diarization vs Clustering

**Diarization (`/transcribe_diarize`):**
- Uses pyannote.audio's pre-trained pipeline
- More accurate for well-separated speakers
- Requires HuggingFace token
- Slower processing time

**Clustering (`/transcribe_cluster`):**
- Uses speaker embeddings (ECAPA/TitaNet) with HDBSCAN clustering
- Better for overlapping speech or noisy audio  
- Faster processing (no full diarization pipeline overhead)
- k-NN post-processing to assign clusters to HDBSCAN noise points (outliers)
- Optional OpenAI name inference using GPT to extract mentioned names
- Automatic embedding visualization with saved PNG plot
- More flexible cluster configuration (min_cluster_size=5, metric=euclidean)

### Embedding Models

The `/transcribe_cluster` endpoint supports three embedding models:

- **ECAPA** (`speechbrain/spkrec-ecapa-voxceleb`): 
  - Fast inference, good general performance
  - 192-dimensional embeddings
  - Best for real-time or resource-constrained scenarios
  
- **TitaNet** (`nvidia/titanet_large`):
  - Higher quality embeddings, slower processing
  - Better speaker discrimination
  - Requires more GPU memory
  
- **Combo**: 
  - Concatenated ECAPA + TitaNet embeddings (both normalized)
  - Combined dimensionality for maximum discriminative power
  - Best accuracy but highest computational cost

All embeddings are L2-normalized before clustering. Models are cached globally after first load.

### Performance Tuning

**Server-side:**
- Adjust `--segment_length` for memory usage (default: 60 seconds, reduce if CUDA OOM occurs)
- Use `--verbose` for detailed timing information and alignment diagnostics
- GPU acceleration is automatic when CUDA is available (model moved to CUDA automatically)
- Modify `TranscriptionTracker` parameters:
  - `min_frequency=4`: Minimum word observations before confirmation (higher = more stable, slower)
  - `min_consecutive_rounds=2`: Consecutive rounds required (higher = fewer false positives)
  - `pos_tolerance=1`: Allowed position drift in tokens (higher = more lenient alignment)
  - `stall_threshold=10`: Rounds before forcing alignment forward (lower = more aggressive)
- Disable progress bars automatically for clean output

**Client-side:**
- Adjust `--send_interval_ms` for latency vs bandwidth (default: 250ms)
  - Lower values = faster response but more network overhead
  - Higher values = slower response but more efficient
- VAD aggressiveness (hardcoded to 3): Higher = more aggressive speech detection
- Silence timeout (hardcoded to 5s): Time before automatic stop

**WebSocket chunk duration:**
- Affects server-side transcription window and confirmation latency
- Larger chunks = more context for ASR but slower word confirmation
- Typical range: 100ms - 500ms

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--segment_length` parameter (try 30 or 45 seconds)
2. **FFmpeg not found**: Ensure you're in the Nix development shell (`nix develop`)
3. **HuggingFace token error**: Set `HUGGINGFACE_ACCESS_TOKEN` in `.env` for diarization endpoint
4. **OpenAI API error**: Set `OPENAI_API_KEY` in `.env` for name inference feature
5. **Audio device issues**: Check microphone permissions and availability (`pactl list sources`)
6. **Typing mode not working**: Install `wtype` for Wayland (`nix-shell -p wtype`)
7. **Words not graduating in streaming**: Increase buffer duration or reduce `min_frequency`/`min_consecutive_rounds`
8. **Streaming diverges from batch**: Use `--stream-test` mode to debug alignment issues
9. **ALSA/JACK errors**: These are suppressed by default; use `--verbose` if debugging audio issues

### Error Handling

- **CLI mode**: Errors printed to console with detailed messages and traceback
- **Server mode**: HTTP error responses with status codes (500 for processing errors) and details
- **WebSocket**: Connection errors handled gracefully with automatic cleanup and client disconnection detection
- **Temporary files**: Automatically cleaned up after processing using Python's `tempfile` and context managers
- **Progress bars**: Suppressed globally using environment variables and stderr redirection
- **Audio extraction**: FFmpeg errors caught and reported with stdout/stderr output
- **Model loading**: Lazy loading to speed up startup; models cached globally after first load

### Advanced Word Stabilization System

The WebSocket streaming endpoint uses a sophisticated multi-layer word confirmation system:

**Layer 1: Frequency Tracking**
- Each word must appear a minimum number of times (default: 4) across transcription rounds
- Prevents spurious one-off ASR hallucinations from appearing in output

**Layer 2: Consecutive Round Observation**
- Words must appear in consecutive transcription rounds (default: 2)
- Ensures words are persistent across ASR outputs, not intermittent

**Layer 3: Position Consistency**
- Words must stabilize at approximately the same token position (tolerance: ±1 token)
- Prevents words that "drift" through the transcript from being confirmed

**Layer 4: Automatic Alignment Recovery**
- Prefix matching: Finds overlap between confirmed words and new transcription
- Partial sequence matching: Falls back to looking for last N words (4→3→2→1)
- Boundary correction: Ensures alignment never moves backward beyond confirmed length
- Forward jump prevention: Limits forward jumps to +1 to avoid skipping unseen tokens

**Layer 5: Rollback Mechanism**
- Detects when ASR appears to have dropped previously confirmed words
- Rolls back a small number of confirmed words (≤3) to maintain alignment
- Moves rolled-back words back to unconfirmed state for re-observation
- Prevents cascading errors when ASR output temporarily loses words

**Layer 6: Stall Detection**
- Tracks when alignment gets stuck at the same position
- Forces alignment forward after threshold (default: 10 rounds) to prevent permanent blocking
- Resets stall counter when alignment successfully advances

**Layer 7: Flexible Matching**
- Number variations: Handles "$3" matching "$3.60" or "three dollars" matching "$3"
- Punctuation normalization: Strips punctuation for matching while preserving in output
- Word number conversion: Maps "one", "two", etc. to "1", "2" for consistent matching
- Currency phrase conversion: "three dollars" → "$3", "five pounds" → "£5"
- Special handling for repeated words appearing at new positions

This system ensures that only truly stable words appear in the output, dramatically reducing false positives while maintaining natural conversation flow.

### Context-Aware Word Tracking (Experimental)

The `context_aware_tracker.py` module provides an alternative tracking system that distinguishes identical words based on their surrounding context:

**How it works:**
- Each word instance is tracked with its context (2 words before, word itself, 1 word after)
- Example: "because like obviously" vs "that like you're" tracks "like" separately in each context
- Creates unique keys: `like@because-like-obviously-that` vs `like@that-like-you're-next`
- Prevents context confusion when the same word appears in different parts of sentences

**Current Status:**
- Implemented in `ContextualWordState` and `ContextAwareTranscriptionTracker` classes
- Integrated into `parakeet.py` with flag `use_context_aware = False` (disabled by default)
- Being tested and validated against standard tracking system
- May be enabled in future versions once thoroughly validated

**Trade-offs:**
- Pro: More accurate for repeated words in different contexts
- Pro: Reduces false matches when words appear in varying sentence structures  
- Con: Requires more observations per contextual instance
- Con: Higher memory usage for tracking contextual keys
