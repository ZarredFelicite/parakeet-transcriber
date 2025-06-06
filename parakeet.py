# How to use:
# In the command line (bash) type (or use python if not python3): python3 transcribe_script.py [audio_filename.wav]

import argparse
import os
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import tempfile
import math
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil # Needed for saving the uploaded file

# Global variable for the ASR model
ASR_MODEL = None

def load_model():
    """Loads the ASR model into the global ASR_MODEL variable if not already loaded."""
    global ASR_MODEL
    if ASR_MODEL is None:
        print("Loading Parakeet TDT model...")
        # Ensure the model is moved to the GPU if available
        ASR_MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2").cuda()
        print("Model loaded and moved to GPU.")
    return ASR_MODEL

import subprocess # For running ffmpeg

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
        raise FileNotFoundError("Error: ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")
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
    audio_for_processing_path = original_input_path # Path to the audio file to process

    try:
        # Ensure the file exists
        if not os.path.exists(original_input_path):
            print(f"Error: Input file not found at {original_input_path}")
            return f"Error: Input file not found at {original_input_path}"

        print(f"Processing input file: {os.path.basename(original_input_path)}")

        # Check file extension to determine if it's a video
        file_extension = os.path.splitext(original_input_path)[1].lower()
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'] # Add more as needed

        if file_extension in video_extensions:
            print(f"Detected video file: {os.path.basename(original_input_path)}. Extracting audio...")
            # Create a temporary directory for the extracted audio
            with tempfile.TemporaryDirectory() as temp_dir_for_extracted_audio:
                 extracted_audio_path = extract_audio_from_video(original_input_path, temp_dir_for_extracted_audio)
                 audio_for_processing_path = extracted_audio_path
                 # The temporary directory will be cleaned up automatically,
                 # but the extracted file path is what we pass to pydub.
                 # pydub might create its own temp files, which are handled below.
                 # We don't need to add extracted_audio_path to temp_files here
                 # because its parent temp_dir_for_extracted_audio is managed by the 'with' statement.
        else:
            print(f"Detected audio file: {os.path.basename(original_input_path)}")
            audio_for_processing_path = original_input_path


        # Load the audio file using pydub
        # Use the extracted audio path if it was a video, otherwise use the original path
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
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        print("Temporary segment files for this transcription cleaned up.")

# --- FastAPI App Definition and Server Mode ---
app = FastAPI()

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

    args = parser.parse_args()

    if args.server:
        load_model() # Load model when server starts
        print(f"Starting Parakeet ASR server with FastAPI/Uvicorn on http://{args.host}:{args.port}")
        # Store segment_length_sec in app.state for access in the endpoint
        app.state.segment_length_sec = args.segment_length
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
