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
import re # Added for websocket_v3 code
import torch  # For device placement in diarization
from pyannote.audio import Pipeline  # Speaker diarization
from dotenv import load_dotenv  # To load HF token from .env if present

# Global variable for the ASR model
ASR_MODEL = None

# Global variable for the diarization pipeline
DIARIZATION_PIPELINE = None

# --- Start of websocket_v3.py content ---

def strip_punctuation(word):
    """Remove punctuation and normalize capitalization for matching purposes"""
    return re.sub(r'[^\\w]', '', word).lower()


def word_to_number(word):
    """Convert word numbers to digits"""
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000', 'million': '1000000'
    }
    return word_to_num.get(word.lower(), word)


def normalize_word_for_matching(word):
    """Normalize word for matching, converting number words to digits and handling currency"""
    # Remove punctuation and convert to lowercase
    clean = strip_punctuation(word)

    # Convert word numbers to digits
    converted = word_to_number(clean)

    return converted


def extract_number_prefix(word):
    """Extract the number portion from a word for flexible number matching"""
    # Match currency symbols, numbers, and common prefixes from original word (before strip_punctuation)
    match = re.match(r'([£$€¥₹]?\\d+)', word.lower())
    return match.group(1) if match else word.lower()


def convert_currency_sequence(words, index):
    """Convert sequences like 'three dollars' to '$3' """
    if index < len(words) - 1:
        current = normalize_word_for_matching(words[index])
        next_word = normalize_word_for_matching(words[index + 1])

        # Check if current is a number and next is currency
        if current.isdigit() and next_word in ['dollar', 'dollars', 'pound', 'pounds', 'euro', 'euros']:
            if next_word in ['dollar', 'dollars']:
                return f"${current}"
            elif next_word in ['pound', 'pounds']:
                return f"£{current}"
            elif next_word in ['euro', 'euros']:
                return f"€{current}"

    return normalize_word_for_matching(words[index])


def words_match_flexible(word1, word2):
    """Check if two words match, with special handling for numbers and currency"""
    # First try exact match after normalization
    norm1 = normalize_word_for_matching(word1)
    norm2 = normalize_word_for_matching(word2)

    if norm1 == norm2:
        return True

    # Try number prefix matching on original words (e.g., "$3" matches "$3.6")
    num1 = extract_number_prefix(word1)
    num2 = extract_number_prefix(word2)

    # Debug number matching
    if num1 != word1.lower() or num2 != word2.lower() or norm1 != word1.lower() or norm2 != word2.lower():
        print(f"[DEBUG NUMBER] Comparing '{word1}' vs '{word2}' | norm: '{norm1}' vs '{norm2}' | nums: '{num1}' vs '{num2}'")

    # If both have number prefixes, check if one is a prefix of the other
    if num1 != word1.lower() and num2 != word2.lower():  # Both contain numbers
        result = num1.startswith(num2) or num2.startswith(num1)
        if result:
            print(f"[DEBUG NUMBER] MATCH: '{word1}' matches '{word2}'")
        return result

    return False


def convert_word_sequence(words):
    """Convert a word sequence, handling currency phrases like 'three dollars' -> '$3'"""
    converted = []
    i = 0
    while i < len(words):
        if i < len(words) - 1:
            current = normalize_word_for_matching(words[i])
            next_word = normalize_word_for_matching(words[i + 1])

            # Check for currency sequence
            if current.isdigit() and next_word in ['dollar', 'dollars', 'pound', 'pounds', 'euro', 'euros']:
                if next_word in ['dollar', 'dollars']:
                    converted.append(f"${current}")
                elif next_word in ['pound', 'pounds']:
                    converted.append(f"£{current}")
                elif next_word in ['euro', 'euros']:
                    converted.append(f"€{current}")
                i += 2  # Skip both words
                continue

        converted.append(normalize_word_for_matching(words[i]))
        i += 1

    return converted


def sequence_matches_flexible(target_seq, words, start_pos):
    """Check if target sequence matches at start_pos with flexible number matching"""
    # Convert both target and transcription to normalized form
    converted_target = convert_word_sequence(target_seq)
    converted_words = convert_word_sequence(words)

    # Adjust start_pos if conversion changed the word array length
    if start_pos >= len(converted_words):
        return False

    if start_pos + len(converted_target) > len(converted_words):
        return False

    # Simple exact matching after conversion
    for i, target_word in enumerate(converted_target):
        current_word = converted_words[start_pos + i]

        # Also try flexible matching for number variations like $3 vs $3.6
        if current_word != target_word and not words_match_flexible(target_word, current_word):
            return False

    return True


class WordState:
    def __init__(self, word, prev_word=None, next_word=None, frequency=1):
        self.word = word  # Original word with punctuation
        self.word_clean = strip_punctuation(word)  # Clean word for matching
        self.prev_word = strip_punctuation(prev_word) if prev_word else None
        self.next_word = strip_punctuation(next_word) if next_word else None
        self.frequency = frequency
        self.state = "unconfirmed"  # "unconfirmed", "potential", "confirmed"
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.latest_form = word  # Track latest punctuation variant

    def increment_frequency(self, word_variant):
        self.frequency += 1
        self.last_seen = time.time()
        self.latest_form = word_variant  # Update to latest punctuation form
        self._update_state()

    def get_context_key(self):
        """Return a key that includes word and context for uniqueness"""
        return (self.word_clean, self.prev_word, self.next_word)

    def _update_state(self):
        if self.frequency >= 4:
            self.state = "confirmed"
        elif self.frequency >= 2:
            self.state = "potential"
        else:
            self.state = "unconfirmed"

    def should_graduate(self):
        return self.state == "confirmed"

    def get_output_word(self):
        """Return the word form to output (with latest punctuation)"""
        return self.latest_form

    def __str__(self):
        context = f"[{self.prev_word or ''}-{self.next_word or ''}]"
        return f"{self.latest_form}({self.frequency}:{self.state}){context}"


class TranscriptionTracker:
    def __init__(self, min_confirmed_words=4):
        self.word_states = {}  # context_key -> WordState
        self.confirmed_words = []  # Simple list of confirmed words (no context)
        self.sent_words_count = 0
        self.min_confirmed_words = min_confirmed_words  # Stop removing words when we reach this many

    def process_transcription(self, words):
        new_words_to_send = []

        # Update word frequencies using context-aware keys
        for i, word in enumerate(words):
            prev_word = words[i-1] if i > 0 else None
            next_word = words[i+1] if i < len(words)-1 else None

            word_state = WordState(word, prev_word, next_word)
            context_key = word_state.get_context_key()

            if context_key in self.word_states:
                # Only increment if not already confirmed (to stop counting at 4)
                existing_state = self.word_states[context_key]
                if existing_state.frequency < 4:
                    existing_state.increment_frequency(word)
                else:
                    # Update latest form but don't increment frequency
                    existing_state.latest_form = word
            else:
                self.word_states[context_key] = word_state

        # Find where our confirmed sequence ends in the current transcription
        start_index = 0
        words_clean = [strip_punctuation(w) for w in words]

        # Search only in the last 20 words to avoid duplicate sequence matches
        search_start = max(0, len(words_clean) - 20)

        # Try cascading search: 4 -> 3 -> 2 -> 1 words
        for n in [4, 3, 2, 1]:
            start_index = self._try_n_word_match(n, words, words_clean, search_start)
            if start_index > 0:
                break

        # If no match found, try fallback with word removal
        if start_index == 0 and len(self.confirmed_words) > 0:
            start_index = self._try_fallback_matching(words, words_clean, search_start)

        if start_index == 0 and len(self.confirmed_words) > 0:
            # This case should ideally be rare if fallback matching works.
            # If it happens, it means we couldn't align the new transcription at all.
            # We might log an error or decide on a recovery strategy,
            # like resetting confirmed_words or sending the whole new transcription.
            # For now, let's print a warning and proceed as if it's a fresh start.
            print(f"Warning: Sequence matching failed completely. Confirmed: {self.confirmed_words[-5:]}, New: {words[:20]}")
            # As a simple recovery, we could reset confirmed_words, but this might lose context.
            # Or, we could try to send the new words if the buffer is small.
            # For now, we'll let it try to graduate from the beginning of `words`.
            # This might lead to re-sending words if `confirmed_words` was not empty.
            # A more robust solution might be needed here depending on observed behavior.
            pass # Allow start_index to remain 0

        # Graduate words in strict sequential order from where confirmed sequence ends
        for i in range(start_index, len(words)):
            word = words[i]
            prev_word = words[i-1] if i > 0 else None
            next_word = words[i+1] if i < len(words)-1 else None

            # Create context key for this specific word instance
            temp_word_state = WordState(word, prev_word, next_word)
            context_key = temp_word_state.get_context_key()

            if context_key in self.word_states:
                word_state = self.word_states[context_key]
                if word_state.should_graduate():
                    # This word can be graduated - use latest punctuation form
                    output_word = word_state.get_output_word()
                    self.confirmed_words.append(output_word)
                    new_words_to_send.append(output_word)
                    self.sent_words_count += 1
                else:
                    # Stop at first non-graduated word to maintain order
                    break
            else:
                # Word context not found, stop graduation
                break

        return new_words_to_send

    def _try_n_word_match(self, n, words, words_clean, search_start, fallback=False):
        """Try to match the last n words from confirmed sequence"""
        if len(self.confirmed_words) < n:
            return 0

        last_n = [strip_punctuation(w) for w in self.confirmed_words[-n:]]
        # prefix = "[DEBUG FALLBACK]" if fallback else "[DEBUG]" # Verbose
        # print(f"{prefix} Looking for last {n}: {last_n} in last 20 words: {words_clean[search_start:]}")

        search_limit = len(words_clean) - n + 1
        for i in range(search_start, search_limit):
            if n == 1:
                # Special case for single word matching
                # Use words[i] for words_match_flexible as it expects original punctuation
                if words_match_flexible(self.confirmed_words[-1], words[i]): # Compare original last confirmed with current original
                    # print(f"{prefix} Found {n}-word match at position {i}, start_index = {i + 1}") # Verbose
                    return i + 1 # Return index in `words` array for next word
            else:
                # Multi-word sequence matching
                # Pass original `words` to sequence_matches_flexible
                if sequence_matches_flexible(self.confirmed_words[-n:], words, i):
                    # print(f"{prefix} Found {n}-word match at position {i}, start_index = {i + n}") # Verbose
                    return i + n # Return index in `words` array for next word
        return 0

    def _try_fallback_matching(self, words, words_clean, search_start):
        """Try matching by progressively removing words from the end of confirmed sequence"""
        removed_words = []
        min_words = getattr(self, 'min_confirmed_words', 4)

        # Keep trying while we have more words than minimum
        while len(self.confirmed_words) >= min_words:
            # print(f"[DEBUG FALLBACK] Removing last confirmed word: '{self.confirmed_words[-1]}'") # Verbose

            # Remove the last confirmed word
            removed_words.append(self.confirmed_words.pop())
            self.sent_words_count -= 1

            # Try cascade: 4->3->2->1 words with remaining confirmed words
            for n in [4, 3, 2, 1]:
                start_index = self._try_n_word_match(n, words, words_clean, search_start, fallback=True)
                if start_index > 0:
                    # print(f"[DEBUG FALLBACK] Found match after removing {len(removed_words)} word(s)") # Verbose
                    return start_index

        # No match found, restore all removed words
        # print(f"[DEBUG FALLBACK] No match found, restoring {len(removed_words)} removed word(s)") # Verbose
        while removed_words:
            self.confirmed_words.append(removed_words.pop())
            self.sent_words_count += 1

        return 0

    def get_debug_info(self):
        confirmed_words_debug = []
        potential_words_debug = []

        # Create a list of (frequency, word_state) for sorting
        sorted_states = sorted(self.word_states.values(), key=lambda ws: ws.frequency, reverse=True)

        for word_state in sorted_states:
            context_str = f"[{word_state.prev_word or ''}-{word_state.next_word or ''}]"
            debug_str = f"{word_state.latest_form}({word_state.frequency}){context_str}"
            if word_state.state == "confirmed" and len(confirmed_words_debug) < 5:
                confirmed_words_debug.append(debug_str)
            elif word_state.state == "potential" and len(potential_words_debug) < 5:
                potential_words_debug.append(debug_str)
            if len(confirmed_words_debug) >=5 and len(potential_words_debug) >=5:
                break

        return {
            "last_5_confirmed": confirmed_words_debug,
            "potential_words": potential_words_debug
        }


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


def process_and_transcribe_audio_file_with_timestamps(input_path: str, segment_length_sec: int = 60):
    """
    Processes an audio (or extracts audio from video) similarly to
    `process_and_transcribe_audio_file` **but** also collects segment–level
    timestamps from NeMo by calling the model with ``timestamps=True``.

    It returns a tuple ``(full_transcription, segments)`` where

    - ``full_transcription`` is a single string containing the entire
      transcription.
    - ``segments`` is a list of dictionaries of the form
      ``{"start": float, "end": float, "text": str}`` where *start* and *end*
      are absolute times in **seconds** from the beginning of the original
      media file.
    """
    asr_model = load_model()

    temp_files = []
    temp_dir = None
    segments_output = []
    all_transcriptions = []

    try:
        # Ensure the source file exists
        if not os.path.exists(input_path):
            return (f"Error: Input file not found at {input_path}", [])

        # Work in a temporary directory so we do not touch the original media
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = temp_dir.name

        file_extension = os.path.splitext(input_path)[1].lower()
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']

        # Extract or copy audio into the temp directory
        if file_extension in video_extensions:
            audio_for_processing_path = extract_audio_from_video(input_path, temp_dir_path)
        else:
            audio_filename = os.path.basename(input_path)
            audio_for_processing_path = os.path.join(temp_dir_path, audio_filename)
            shutil.copyfile(input_path, audio_for_processing_path)

        # Load and normalise with pydub
        audio = AudioSegment.from_file(audio_for_processing_path)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)

        # Segment the audio to keep memory usage bounded
        segment_length_ms = segment_length_sec * 1000
        total_length_ms = len(audio)
        num_segments = math.ceil(total_length_ms / segment_length_ms)

        for i in range(num_segments):
            start_time_ms = i * segment_length_ms
            end_time_ms = min((i + 1) * segment_length_ms, total_length_ms)

            segment = audio[start_time_ms:end_time_ms]

            # Export segment to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=f"_{i}.wav", delete=False) as tmpfile:
                temp_wav_path = tmpfile.name
                temp_files.append(temp_wav_path)
            segment.export(temp_wav_path, format='wav')

            # Transcribe with timestamps
            transcription_result = asr_model.transcribe([temp_wav_path], timestamps=True)

            if transcription_result and len(transcription_result) > 0:
                sample = transcription_result[0]
                # Collect raw text
                if hasattr(sample, 'text') and sample.text:
                    all_transcriptions.append(sample.text)
                else:
                    all_transcriptions.append("")

                # Collect segment-level timestamps (if provided by NeMo)
                if hasattr(sample, 'timestamp') and 'segment' in sample.timestamp:
                    for ts in sample.timestamp['segment']:
                        segments_output.append({
                            "start": ts['start'] + (start_time_ms / 1000.0),
                            "end": ts['end'] + (start_time_ms / 1000.0),
                            "text": ts['segment']
                        })
            else:
                # Keep position in final transcription even if this chunk fails
                all_transcriptions.append("[Transcription Failed for Segment]")

        final_transcription = " ".join(all_transcriptions)
        return (final_transcription, segments_output)

    except Exception as e:
        error_msg = f"An error occurred during audio processing: {type(e).__name__}: {e}"
        return (error_msg, [])
    finally:
        # Clean up temporary WAV segment files
        for p in temp_files:
            if os.path.exists(p):
                os.remove(p)
        # Clean up the temporary directory
        if temp_dir:
            temp_dir.cleanup()


# --- FastAPI App Definition and Server Mode ---
app = FastAPI()

@app.websocket("/ws/transcribe")
async def websocket_transcribe_endpoint(websocket: WebSocket):
    asr_model = load_model()
    # Pass app.state.verbose to the underlying v3 endpoint
    verbose_logging = app.state.verbose if hasattr(app.state, 'verbose') else False
    # The logic from the original websocket_transcribe_v3_endpoint is now directly here.
    # We need to ensure all logic from the original websocket_transcribe_v3_endpoint is correctly placed.
    # This means the function signature changes and it directly handles the WebSocket logic.

    await websocket.accept() # This was in the original _v3_endpoint

    # Get chunk duration from query params
    try:
        chunk_duration_ms = int(websocket.query_params.get("chunk_duration_ms", 250))
    except (TypeError, ValueError):
        chunk_duration_ms = 250

    receive_timeout = (chunk_duration_ms / 1000.0) * 2 # Wait for at least two chunks

    # Constants
    sample_rate = 16000
    sample_width = 2 # 16-bit
    channels = 1
    window_size_seconds = 15 # Max audio buffer for transcription
    min_audio_len_seconds = 5 # Min audio for initial transcription
    min_audio_len_bytes = min_audio_len_seconds * sample_rate * sample_width * channels
    window_size_bytes = window_size_seconds * sample_rate * sample_width * channels

    buffer = bytearray()
    tracker = TranscriptionTracker()
    is_initial_transcription = True

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=receive_timeout)
                buffer.extend(data)
            except asyncio.TimeoutError:
                # If timeout, and we have some data, proceed to transcribe
                if not buffer:
                    if verbose_logging:
                        print(f"[{time.strftime('%H:%M:%S')}] Timeout with no buffer, continuing.")
                    continue # No data and timeout, just continue waiting
                if verbose_logging:
                    print(f"[{time.strftime('%H:%M:%S')}] Timeout, processing buffered audio: {len(buffer)} bytes")
                data = None # Indicate timeout processing

            # Determine audio data for transcription
            current_buffer_len = len(buffer)
            audio_to_transcribe_data = None

            if is_initial_transcription:
                if current_buffer_len >= min_audio_len_bytes:
                    audio_to_transcribe_data = buffer
                    is_initial_transcription = False # Met initial length, subsequent transcriptions are not "initial"
                    if verbose_logging:
                        print(f"[{time.strftime('%H:%M:%S')}] Initial transcription with {current_buffer_len} bytes.")
                else:
                    if not data and current_buffer_len > 0: # Timeout with some data but not enough for initial
                        if verbose_logging:
                            print(f"[{time.strftime('%H:%M:%S')}] Timeout with insufficient data for initial: {current_buffer_len} bytes. Waiting for more.")
                        continue
                    else: # Still accumulating data for initial transcription
                        if verbose_logging:
                            print(f"[{time.strftime('%H:%M:%S')}] Accumulating initial data: {current_buffer_len}/{min_audio_len_bytes} bytes.")
                        continue
            else: # Not initial transcription
                if current_buffer_len == 0: # No new data
                    if verbose_logging:
                        print(f"[{time.strftime('%H:%M:%S')}] No new data and not initial, continuing.")
                    continue
                # Use the entire buffer up to window_size_bytes
                # The buffer itself is managed to not exceed window_size_bytes later
                audio_to_transcribe_data = buffer

            if not audio_to_transcribe_data: # Should not happen if logic above is correct, but as a safeguard
                if verbose_logging:
                    print(f"[{time.strftime('%H:%M:%S')}] No audio data to transcribe, continuing.")
                continue

            # Convert raw bytes to AudioSegment
            audio_segment = AudioSegment(data=audio_to_transcribe_data, sample_width=sample_width, frame_rate=sample_rate, channels=channels)

            # Transcribe audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                temp_wav_file_path = tmpfile.name
                audio_segment.export(temp_wav_file_path, format="wav")

            transcription_start_time = time.time()
            transcription_result = asr_model.transcribe([temp_wav_file_path])
            transcription_end_time = time.time()
            os.remove(temp_wav_file_path)

            current_words = transcription_result[0].text.strip().split() if transcription_result and transcription_result[0].text else []

            if verbose_logging:
                print(f"[{time.strftime('%H:%M:%S')}] Audio: {len(audio_segment)/1000:.2f}s, Transcribe time: {transcription_end_time-transcription_start_time:.2f}s, Raw: {' '.join(current_words)}")

            # Process through WordState tracker
            new_words_to_send = tracker.process_transcription(current_words)

            if new_words_to_send:
                await websocket.send_text(" ".join(new_words_to_send))
                if verbose_logging:
                    print(f"Sent: {' '.join(new_words_to_send)}")


            # Manage buffer: if it's longer than window_size_bytes, keep only the latest part
            if len(buffer) > window_size_bytes:
                buffer = buffer[-window_size_bytes:]
            elif is_initial_transcription == False and not data : # If it was a timeout and not initial, clear buffer as it was processed
                buffer = bytearray()


    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.") # Updated message slightly
    except Exception as e:
        print(f"An error occurred in WebSocket: {e}") # Updated message slightly
        import traceback # Ensure traceback is imported if not already global
        traceback.print_exc()
    finally:
        if websocket.client_state.value != 3: # Check if not already closed (WS_STATE_CLOSED = 3)
             await websocket.close()


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


@app.post("/transcribe_timestamps")
async def transcribe_timestamps_endpoint(audio_file: UploadFile = File(...)):
    """
    Upload an audio **or** video file and receive the transcription *along with*
    absolute segment-level timestamps produced by the NeMo model.

    Response JSON schema::
        {
            "transcription": "full transcription as a single string",
            "segments": [
                {"start": 0.0, "end": 4.23, "text": "first spoken segment"},
                ...
            ]
        }
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, audio_file.filename)
            with open(temp_input_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            # Use the segment length configured for the application (default 60)
            segment_length_sec = getattr(app.state, 'segment_length_sec', 60)

            transcription, segments = process_and_transcribe_audio_file_with_timestamps(
                temp_input_path,
                segment_length_sec,
            )

            if transcription.startswith("Error:") or "[Transcription Failed for Segment]" in transcription:
                return JSONResponse(
                    {"error": "Transcription failed or error during processing", "details": transcription},
                    status_code=500,
                )

            return JSONResponse({"transcription": transcription, "segments": segments})

    except Exception as e:
        print(f"Error handling transcription request for {audio_file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


def load_diarization_pipeline():
    """Load (or return cached) speaker-diarization pipeline."""
    global DIARIZATION_PIPELINE
    if DIARIZATION_PIPELINE is None:
        # Ensure we have the token (either already in env or loaded from .env)
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")
        if hf_token == "":
            raise EnvironmentError(
                "HUGGINGFACE_ACCESS_TOKEN not set in environment or .env file; required for diarization pipeline."  # noqa: E501
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading pyannote speaker-diarization pipeline on {device}…")
        DIARIZATION_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        DIARIZATION_PIPELINE.to(device)
    return DIARIZATION_PIPELINE


# ---------------------------------------------------------------------------
# Helper to tag ASR segments with speaker labels
# ---------------------------------------------------------------------------

def annotate_segments_with_speakers(asr_segments, diarization_result):
    """Return list of ASR segments augmented with a *speaker* key.

    Parameters
    ----------
    asr_segments : list[dict]
        Each dict must have ``start`` & ``end`` times (seconds) plus ``text``.
    diarization_result : pyannote.core.Annotation
        Result returned by ``Pipeline``.
    Returns
    -------
    list[dict]
        Same length as *asr_segments* with an extra ``speaker`` entry.
    """
    # Build list of diarization segments with speaker labels for quick overlap calc
    diar_segments = []
    for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
        diar_segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker_label),
            }
        )

    annotated = []
    for seg in asr_segments:
        # Compute total overlap per speaker
        overlaps = {}
        for d in diar_segments:
            start = max(seg["start"], d["start"])
            end = min(seg["end"], d["end"])
            if end > start:  # positive overlap
                overlaps[d["speaker"]] = overlaps.get(d["speaker"], 0.0) + (end - start)
        # Pick speaker with max overlap (>0) else unknown
        if overlaps:
            speaker = max(overlaps, key=overlaps.get)
        else:
            speaker = "unknown"
        annotated.append({**seg, "speaker": speaker})
    return annotated


def join_consecutive_segments_by_speaker(annotated_segments):
    """Merge consecutive segments by speaker while preserving individual segments.

    Each element in the returned list contains an additional key ``segments``
    which is a list of the original (un-merged) ASR segments belonging to that
    speaker block.
    """
    if not annotated_segments:
        return []

    # Start with the first segment and initialise its list of sub-segments.
    first = annotated_segments[0].copy()
    first["segments"] = [annotated_segments[0].copy()]
    joined = [first]

    for seg in annotated_segments[1:]:
        last = joined[-1]
        if seg["speaker"] == last["speaker"]:
            # Same speaker – extend end time, concatenate text, and store segment.
            last["end"] = seg["end"]
            last["text"] = (last["text"] + " " + seg["text"].strip()).strip()
            last["segments"].append(seg.copy())
        else:
            new_entry = seg.copy()
            new_entry["segments"] = [seg.copy()]
            joined.append(new_entry)

    return joined


@app.post("/transcribe_diarize")
async def transcribe_diarize_endpoint(audio_file: UploadFile = File(...)):
    """Transcribe + diarize audio/video and tag each segment with the speaker.

    Response example::
        {
            "segments": [
                {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00", "text": "Hello there."},
                {"start": 3.2, "end": 7.5, "speaker": "SPEAKER_01", "text": "Hi!"}
            ]
        }
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # ------------------------------------------------------------------
            # Save uploaded file
            # ------------------------------------------------------------------
            temp_input_path = os.path.join(temp_dir, audio_file.filename)
            with open(temp_input_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            # ------------------------------------------------------------------
            # If video, extract audio for both ASR & diarization
            # ------------------------------------------------------------------
            file_extension = os.path.splitext(temp_input_path)[1].lower()
            video_ext = [
                ".mp4",
                ".mkv",
                ".avi",
                ".mov",
                ".wmv",
                ".flv",
                ".webm",
            ]
            if file_extension in video_ext:
                diar_audio_path = extract_audio_from_video(temp_input_path, temp_dir)
            else:
                diar_audio_path = temp_input_path

            # ------------------------------------------------------------------
            # Ensure diar_audio_path is mono-16k WAV (required by pyannote)
            # ------------------------------------------------------------------
            diar_audio_path = convert_audio_to_wav_mono_16k(diar_audio_path, temp_dir)

            # ------------------------------------------------------------------
            # ASR with timestamps (re-uses existing helper)
            # ------------------------------------------------------------------
            segment_length_sec = getattr(app.state, "segment_length_sec", 60)
            transcription, asr_segments = process_and_transcribe_audio_file_with_timestamps(
                diar_audio_path, segment_length_sec
            )

            if transcription.startswith("Error:"):
                return JSONResponse(
                    {"error": "ASR processing failed", "details": transcription},
                    status_code=500,
                )

            # ------------------------------------------------------------------
            # Diarization
            # ------------------------------------------------------------------
            diar_pipeline = load_diarization_pipeline()
            diar_result = diar_pipeline(diar_audio_path)

            # ------------------------------------------------------------------
            # Annotate & join segments
            # ------------------------------------------------------------------
            annotated = annotate_segments_with_speakers(asr_segments, diar_result)
            joined_segments = join_consecutive_segments_by_speaker(annotated)

            return JSONResponse({"segments": joined_segments})

    except Exception as e:
        print(f"Error in /transcribe_diarize endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def convert_audio_to_wav_mono_16k(src_path: str, dst_dir: str) -> str:
    """Convert *src_path* audio to a temporary mono 16-kHz WAV inside *dst_dir*.

    Returns the path to the new WAV file. If *src_path* is already a WAV with
    single channel and 16 kHz, the file is just copied to the destination.
    """
    # Determine destination path
    base_name = os.path.splitext(os.path.basename(src_path))[0]
    dst_path = os.path.join(dst_dir, f"{base_name}_16k_mono.wav")

    try:
        audio = AudioSegment.from_file(src_path)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        audio.export(dst_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"Failed to convert audio to WAV: {e}")

    return dst_path


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
