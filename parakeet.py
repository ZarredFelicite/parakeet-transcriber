import argparse
import os
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import tempfile
import math
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import shutil # Needed for saving the uploaded file
import subprocess # For running ffmpeg
import numpy as np
import difflib
import time
from collections import deque
import asyncio
from websocket_v3 import websocket_transcribe_v3_endpoint

# Global variable for the ASR model
ASR_MODEL = None

class WordState:
    def __init__(self, word, state="unconfirmed", stable_count=0):
        self.word = word
        self.state = state # "unconfirmed", "potential", "confirmed"
        self.stable_count = stable_count # How many consecutive times it appeared after the anchor

def find_longest_common_substring(list_of_lists):
    """Finds the longest common sequence of items anywhere in several lists."""
    if not list_of_lists:
        return []

    shortest_list = min(list_of_lists, key=len)
    longest_common_substring = []

    for i in range(len(shortest_list)):
        for j in range(i, len(shortest_list)):
            substring = shortest_list[i:j+1]
            is_common = all(
                difflib.SequenceMatcher(None, substring, other_list).find_longest_match(0, len(substring), 0, len(other_list)).size == len(substring)
                for other_list in list_of_lists
            )
            if is_common and len(substring) > len(longest_common_substring):
                longest_common_substring = substring

    return longest_common_substring

def check_ffmpeg_in_path():
    """Checks if ffmpeg is available in the system's PATH."""
    print("Checking for ffmpeg in PATH...")
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
        print("ffmpeg found in PATH.")
    except FileNotFoundError:
        raise FileNotFoundError("Error: ffmpeg not found in your system's PATH. Please install ffmpeg to process video files.")
    except Exception as e:
        print(f"An unexpected error occurred while checking for ffmpeg: {e}")
        # Still raise FileNotFoundError for consistency with the intended check failure
        raise FileNotFoundError("Error: Could not verify ffmpeg in PATH. Please ensure ffmpeg is installed and accessible.")


def load_model():
    """Loads the ASR model into the global ASR_MODEL variable if not already loaded."""
    global ASR_MODEL
    if ASR_MODEL is None:
        check_ffmpeg_in_path() # Check for ffmpeg before loading the model
        print("Loading Parakeet TDT model...")
        # Ensure the model is moved to the GPU if available
        ASR_MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2").cuda()
        print("Model loaded and moved to GPU.")
    return ASR_MODEL


def extract_audio_from_video(video_path: str, output_dir: str) -> str:
    """
    Extracts audio from a video file using ffmpeg and saves it as a WAV file.
    Args:
        video_path (str): The path to the video file.
        output_dir (str): The directory to save the extracted audio file.
    Returns:
        str: The path to the extracted WAV audio file.
    Raises:
        Exception: If ffmpeg command fails.
    """
    base_name = os.path.basename(video_path)
    audio_filename = f"{os.path.splitext(base_name)[0]}.wav"
    output_audio_path = os.path.join(output_dir, audio_filename)

    # ffmpeg command to extract audio and convert to WAV (mono, 16kHz)
    # -i: input file
    # -vn: no video
    # -acodec pcm_s16le: audio codec (16-bit signed little-endian PCM)
    # -ac 1: audio channels (mono)
    # -ar 16000: audio sample rate (16kHz)
    # -y: overwrite output file without asking
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "-y",
        output_audio_path
    ]

    print(f"Extracting audio from video: {video_path}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Audio extracted to: {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e}")
        print(f"ffmpeg stdout: {e.stdout}")
        print(f"ffmpeg stderr: {e.stderr}")
        raise Exception(f"ffmpeg audio extraction failed: {e.stderr}")
    except FileNotFoundError:
        # This specific FileNotFoundError is for the ffmpeg command itself within subprocess.run
        # The check_ffmpeg_in_path should ideally catch this earlier, but keeping this
        # for robustness in case the PATH changes or ffmpeg becomes unavailable later.
        raise FileNotFoundError("Error: ffmpeg not found during extraction. Please ensure ffmpeg is installed and in your system's PATH.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during audio extraction: {e}")


def process_and_transcribe_audio_file(input_path: str, segment_length_sec: int = 60) -> str:
    """
    Processes an audio file (or extracts audio from video), ensuring it's mono and 16kHz,
    splits it into segments, transcribes them, and returns the full transcription.
    Assumes the ASR model is already loaded via load_model().
    Args:
        input_path (str): The path to the audio or video file.
        segment_length_sec (int): The length of each audio segment in seconds.
    Returns:
        str: The full transcription or an error message.
    """
    asr_model = load_model() # Ensures model is loaded and gets the instance
    original_input_path = input_path # Keep track of the original input path
    temp_files = [] # To keep track of temporary segment files
    all_transcriptions = [] # Initialize list to store transcriptions of segments
    temp_dir = None # Initialize temp_dir to None

    try:
        # Ensure the file exists
        if not os.path.exists(original_input_path):
            print(f"Error: Input file not found at {original_input_path}")
            return f"Error: Input file not found at {original_input_path}"

        print(f"Processing input file: {os.path.basename(original_input_path)}")

        # Create a temporary directory for extracted audio and segments
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = temp_dir.name

        # Check file extension to determine if it's a video
        file_extension = os.path.splitext(original_input_path)[1].lower()
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'] # Add more as needed

        if file_extension in video_extensions:
            print(f"Detected video file: {os.path.basename(original_input_path)}. Extracting audio...")
            audio_for_processing_path = extract_audio_from_video(original_input_path, temp_dir_path)
        else:
            print(f"Detected audio file: {os.path.basename(original_input_path)}")
            # If it's an audio file, copy it to the temporary directory for consistent handling
            audio_filename = os.path.basename(original_input_path)
            audio_for_processing_path = os.path.join(temp_dir_path, audio_filename)
            shutil.copyfile(original_input_path, audio_for_processing_path)


        # Load the audio file using pydub
        audio = AudioSegment.from_file(audio_for_processing_path)

        # Check and convert to mono if necessary
        if audio.channels > 1:
            print(f"Audio has {audio.channels} channels. Converting to mono.")
            audio = audio.set_channels(1)

        # Check sample rate and resample to 16kHz if necessary (standard for ASR models)
        if audio.frame_rate != 16000:
            print(f"Audio has sample rate {audio.frame_rate} Hz. Resampling to 16000 Hz.")
            audio = audio.set_frame_rate(16000)

        # --- Split audio into segments ---
        segment_length_ms = segment_length_sec * 1000
        total_length_ms = len(audio)
        num_segments = math.ceil(total_length_ms / segment_length_ms)

        print(f"Total audio length: {total_length_ms / 1000:.2f} seconds")
        print(f"Splitting into {num_segments} segments of up to {segment_length_sec} seconds.")

        # all_transcriptions list is initialized at the beginning of the function.

        for i in range(num_segments):
            start_time = i * segment_length_ms
            end_time = min((i + 1) * segment_length_ms, total_length_ms)
            segment = audio[start_time:end_time]

            # Create a temporary WAV file for the segment
            with tempfile.NamedTemporaryFile(suffix=f'_{i}.wav', delete=False) as tmpfile:
                temp_wav_file_path = tmpfile.name
                temp_files.append(temp_wav_file_path) # Add to list for cleanup

            # Export the segment to the temporary file
            segment.export(temp_wav_file_path, format='wav')
            # print(f"Exported segment {i+1}/{num_segments} to {temp_wav_file_path}") # Uncomment for debugging

            # Transcribe the current segment
            print(f"Transcribing segment {i+1}/{num_segments} ({start_time/1000:.2f}s - {end_time/1000:.2f}s)...")
            segment_transcription_result = asr_model.transcribe([temp_wav_file_path])

            if segment_transcription_result and len(segment_transcription_result) > 0 and hasattr(segment_transcription_result[0], 'text'):
                all_transcriptions.append(segment_transcription_result[0].text)
            else:
                print(f"Warning: Transcription failed or returned unexpected format for segment {i+1}.")
                all_transcriptions.append("[Transcription Failed for Segment]")

        final_transcription = " ".join(all_transcriptions)
        return final_transcription

    except Exception as e:
        error_msg = f"An error occurred during audio processing: {type(e).__name__}: {e}"
        print(error_msg)
        return error_msg
    finally:
        # Clean up temporary segment files
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        print("Temporary segment files for this transcription cleaned up.")
        # Clean up the main temporary directory
        if temp_dir:
            temp_dir.cleanup()
            print(f"Temporary directory {temp_dir_path} cleaned up.")


# --- FastAPI App Definition and Server Mode ---
app = FastAPI()

@app.websocket("/ws/transcribe")
async def websocket_transcribe_endpoint(websocket: WebSocket):
    await websocket.accept()
    asr_model = load_model()
    
    # Constants for audio processing
    sample_rate = 16000
    sample_width = 2  # 16-bit
    channels = 1
    
    # Buffer settings
    window_size_seconds = 5
    step_size_seconds = 1
    window_size_bytes = window_size_seconds * sample_rate * sample_width
    step_size_bytes = step_size_seconds * sample_rate * sample_width

    buffer = bytearray()
    last_transcription = ""
    
    try:
        is_initial_transcription = True
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)

            if is_initial_transcription:
                # For the first transcription, pad with silence to 5 seconds
                if len(buffer) < window_size_bytes:
                    silence_duration_ms = (window_size_bytes - len(buffer)) * 1000 // (sample_rate * sample_width)
                    padding = AudioSegment.silent(duration=silence_duration_ms, frame_rate=sample_rate)
                    audio_data = padding + AudioSegment(data=buffer, sample_width=sample_width, frame_rate=sample_rate, channels=channels)
                else:
                    audio_data = AudioSegment(data=buffer[:window_size_bytes], sample_width=sample_width, frame_rate=sample_rate, channels=channels)
                    is_initial_transcription = False
            else:
                # Use a sliding window after the initial transcription
                if len(buffer) < window_size_bytes:
                    continue
                audio_data = AudioSegment(data=buffer[:window_size_bytes], sample_width=sample_width, frame_rate=sample_rate, channels=channels)


            # Create a temporary file for the audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                temp_wav_file_path = tmpfile.name
                audio_data.export(temp_wav_file_path, format="wav")

            # Transcribe the audio window
            transcription_result = asr_model.transcribe([temp_wav_file_path])
            os.remove(temp_wav_file_path)

            current_transcription = ""
            if transcription_result and len(transcription_result) > 0:
                current_transcription = transcription_result[0].text.strip()

            if app.state.verbose:
                print(f"Segment Transcription: {current_transcription}")

            # More robust segment matching
            new_words = []
            if last_transcription:
                # Find the longest common subsequence
                s = difflib.SequenceMatcher(None, last_transcription.split(), current_transcription.split())
                match = s.find_longest_match(0, len(last_transcription.split()), 0, len(current_transcription.split()))
                
                if match.size > 0:
                    # If there's a common subsequence, append the new words after it
                    new_words = current_transcription.split()[match.b + match.size:]
                else:
                    # If no common subsequence, send the whole thing (fallback)
                    new_words = current_transcription.split()
            else:
                # First transcription
                new_words = current_transcription.split()

            if new_words:
                await websocket.send_text(" ".join(new_words))

            last_transcription = current_transcription
            
            # Slide the buffer window
            buffer = buffer[step_size_bytes:]

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.")
    except Exception as e:
        print(f"An error occurred in WebSocket: {e}")
    finally:
        await websocket.close()

@app.websocket("/ws/transcribe_v2")
async def websocket_transcribe_v2_endpoint(websocket: WebSocket):
    await websocket.accept()
    asr_model = load_model()

    # Get chunk duration from query params, with a default
    try:
        chunk_duration_ms = int(websocket.query_params.get("chunk_duration_ms", 250))
    except (TypeError, ValueError):
        chunk_duration_ms = 250
    
    receive_timeout = (chunk_duration_ms / 1000.0) * 2

    # Constants
    sample_rate = 16000
    sample_width = 2
    channels = 1
    window_size_seconds = 15
    min_audio_len_bytes = 5 * sample_rate * sample_width
    window_size_bytes = window_size_seconds * sample_rate * sample_width

    buffer = bytearray()
    transcription_history = deque(maxlen=3)
    confirmed_words = []
    potential_words_history = deque(maxlen=3)
    anchor_misses = 0
    is_initial_transcription = True

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=receive_timeout)
                buffer.extend(data)
            except asyncio.TimeoutError:
                pass

            if is_initial_transcription:
                if len(buffer) >= min_audio_len_bytes:
                    is_initial_transcription = False
                
                silence_duration_ms = (min_audio_len_bytes - len(buffer)) * 1000 // (sample_rate * sample_width)
                padding = AudioSegment.silent(duration=silence_duration_ms, frame_rate=sample_rate)
                audio_data = padding + AudioSegment(data=buffer, sample_width=sample_width, frame_rate=sample_rate, channels=channels)
            else:
                if len(buffer) > window_size_bytes:
                    buffer = buffer[-window_size_bytes:]
                audio_data = AudioSegment(data=buffer, sample_width=sample_width, frame_rate=sample_rate, channels=channels)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                temp_wav_file_path = tmpfile.name
                audio_data.export(temp_wav_file_path, format="wav")
            
            start_time = time.time()
            transcription_result = asr_model.transcribe([temp_wav_file_path])
            end_time = time.time()
            os.remove(temp_wav_file_path)

            current_words = transcription_result[0].text.strip().split() if transcription_result and transcription_result[0].text else []
            
            if app.state.verbose:
                print(f"[{time.strftime('%H:%M:%S')}] Audio: {len(audio_data)/1000:.2f}s, Transcribe time: {end_time-start_time:.2f}s, Raw: {' '.join(current_words)}")

            transcription_history.append(current_words)

            if not confirmed_words:
                # Initial stage
                if len(transcription_history) == 3:
                    stable_string = find_longest_common_substring(list(transcription_history))
                    if stable_string:
                        confirmed_words.extend(stable_string)
                        await websocket.send_text(" ".join(confirmed_words))
                        if app.state.verbose:
                            print(f"Confirmed initial words: {' '.join(confirmed_words)}")
            else:
                # Anchor stage
                anchor_size = min(len(confirmed_words), 2)
                anchor = confirmed_words[-anchor_size:]
                if app.state.verbose:
                    print(f"Using anchor: {' '.join(anchor)}")
                anchor_found = False
                try:
                    anchor_index = current_words.index(anchor[0], 0)
                    while anchor_index != -1:
                        if current_words[anchor_index:anchor_index+len(anchor)] == anchor:
                            potential_new_words = current_words[anchor_index + len(anchor):]
                            if app.state.verbose:
                                print(f"Found anchor. Potential new words: {' '.join(potential_new_words)}")
                            potential_words_history.append(potential_new_words)
                            anchor_found = True
                            break
                        anchor_index = current_words.index(anchor[0], anchor_index + 1)
                except ValueError:
                    pass # Anchor not found

                if anchor_found:
                    anchor_misses = 0
                else:
                    anchor_misses += 1
                    if app.state.verbose:
                        print(f"Anchor not found. Misses: {anchor_misses}")

                if anchor_misses >= 3:
                    if app.state.verbose:
                        print("Three consecutive anchor misses. Falling back to substring matching.")
                    potential_words_history.clear()
                    # Fallback to substring matching
                    if len(transcription_history) == 3:
                        stable_string = find_longest_common_substring(list(transcription_history))
                        if app.state.verbose:
                            print(f"Fallback stable string: {' '.join(stable_string)}")
                        if stable_string and stable_string != confirmed_words:
                            # Check for continuation
                            if len(stable_string) > len(confirmed_words) and stable_string[:len(confirmed_words)] == confirmed_words:
                                new_words = stable_string[len(confirmed_words):]
                                if app.state.verbose:
                                    print(f"Fallback continuation. New words: {' '.join(new_words)}")
                                confirmed_words.extend(new_words)
                                await websocket.send_text(" ".join(new_words))
                            else:
                                # Correction
                                if app.state.verbose:
                                    print(f"Fallback correction. New transcription: {' '.join(stable_string)}")
                                confirmed_words = stable_string
                                await websocket.send_text(" ".join(confirmed_words))
                    anchor_misses = 0 # Reset after fallback
                elif len(potential_words_history) == 3:
                    stable_new_words = find_longest_common_substring(list(potential_words_history))
                    if app.state.verbose:
                        print(f"Checking for stable new words: {' '.join(stable_new_words)}")
                    if stable_new_words:
                        if app.state.verbose:
                            print(f"Confirmed new words: {' '.join(stable_new_words)}")
                        await websocket.send_text(" ".join(stable_new_words))
                        confirmed_words.extend(stable_new_words)
                        potential_words_history.clear()

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket V2.")
    except Exception as e:
        print(f"An error occurred in WebSocket V2: {e}")
    finally:
        if websocket.client_state.value != 3:
            await websocket.close()

@app.websocket("/ws/transcribe_v3")
async def websocket_transcribe_v3_endpoint_wrapper(websocket: WebSocket):
    asr_model = load_model()
    await websocket_transcribe_v3_endpoint(websocket, asr_model, verbose=app.state.verbose if hasattr(app.state, 'verbose') else False)


@app.post("/transcribe")
async def transcribe_endpoint(audio_file: UploadFile = File(...)):
    """
    Receives an audio or video file via POST request and returns the transcription.
    """
    # FastAPI handles file uploads differently.
    # We need to save the uploaded file to a temporary location.
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, audio_file.filename)
            with open(temp_input_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            print(f"Received file for transcription: {audio_file.filename}")

            # Access segment_length_sec from app.state
            segment_length_sec = app.state.segment_length_sec
            # Pass the temporary input path to the processing function
            transcription = process_and_transcribe_audio_file(temp_input_path, segment_length_sec)

            if transcription.startswith("Error:") or "[Transcription Failed for Segment]" in transcription:
                 return JSONResponse({"error": "Transcription failed or error during processing", "details": transcription}, status_code=500)
            return JSONResponse({"transcription": transcription})
    except Exception as e:
        print(f"Error handling transcription request for {audio_file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Parakeet TDT or run as a transcription server.")

    parser.add_argument("audio_file", nargs='?', default=None,
                        help="Path to the audio file (WAV or MP3) for CLI mode. Not used if --server is specified.")

    parser.add_argument("--segment_length", type=int, default=60,
                        help="Length of audio segments in seconds (default: 60). "
                             "Applies to both CLI and server mode processing. "
                             "Decrease if you encounter out-of-memory errors.")

    parser.add_argument("--server", action="store_true",
                        help="Run in server mode. If this flag is set, 'audio_file' argument is ignored.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host for the server (default: '0.0.0.0'). Only used with --server.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port for the server (default: 5000). Only used with --server.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for debugging.")

    args = parser.parse_args()

    if args.server:
        load_model() # Load model when server starts
        print(f"Starting Parakeet ASR server with FastAPI/Uvicorn on http://{args.host}:{args.port}")
        # Store segment_length_sec in app.state for access in the endpoint
        app.state.segment_length_sec = args.segment_length
        app.state.verbose = args.verbose
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        if args.audio_file is None:
            parser.error("The 'audio_file' argument is required when not running in --server mode.")

        print(f"Running in CLI mode for input file: {args.audio_file}")
        # process_and_transcribe_audio_file calls load_model() internally.
        final_transcription = process_and_transcribe_audio_file(args.audio_file, args.segment_length)

        if final_transcription.startswith("Error:") or "[Transcription Failed for Segment]" in final_transcription:
            print(f"\nCLI Transcription Failed/Error: {final_transcription}")
        else:
            print("\nFull Transcription:")
            print(final_transcription)

        print("CLI transcription process complete.")
