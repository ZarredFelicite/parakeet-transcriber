import argparse
import os
import tempfile
import math
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
import shutil  # Needed for saving the uploaded file
import subprocess  # For running ffmpeg
import numpy as np
import time
import asyncio
import re  # For websocket logic
import uuid
import json

# Lightweight imports kept at module load
from pydub import AudioSegment

# Heavy / optional dependencies are imported lazily inside functions:
# - nemo.collections.asr (NeMo)
# - torch
# - pyannote.audio (Pipeline, Audio, PretrainedSpeakerEmbedding, Segment)
# - hdbscan
# - matplotlib / TSNE / PCA
# - soundfile (sf)
# - sklearn.neighbors (NearestNeighbors)
# - requests (OpenAI API)
# - python-dotenv (load_dotenv)
# Each function that needs them will import locally so that simple server startup
# or using only basic endpoints is faster.

# --- Disable tqdm progress bars globally (NeMo transcribe uses tqdm) ---
# This prevents the persistent "Transcribing: 100%|" bars from cluttering stdout.
# Use environment variables only to avoid monkey-patching recursion issues.
import os as _os
_os.environ.setdefault("TQDM_DISABLE", "1")
_os.environ.setdefault("DISABLE_NEMO_PROGRESS_BAR", "1")

# Additional progress bar suppression
try:
    import sys
    from io import StringIO
    _original_stderr = sys.stderr
    def _suppress_progress_bars():
        sys.stderr = StringIO()
    def _restore_stderr():
        sys.stderr = _original_stderr
except Exception:
    def _suppress_progress_bars(): pass
    def _restore_stderr(): pass

# Global variable for the ASR model
ASR_MODEL = None

# Global variable for the diarization pipeline
DIARIZATION_PIPELINE = None

# Cache of speaker-embedding models keyed by name
EMBEDDING_MODELS = {}

# --- Start of websocket_v3.py content ---

def strip_punctuation(word):
    """Remove punctuation and normalize capitalization for matching purposes"""
    return re.sub(r'[^\w\s]', '', word).lower()


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


class WordState:
    def __init__(self, word, frequency=1, initial_round=None):
        self.word = word  # Original word with punctuation
        self.word_clean = normalize_word_for_matching(word)  # Normalized for robust matching
        self.frequency = frequency
        self.state = "unconfirmed"  # "unconfirmed", "potential", "confirmed"
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.latest_form = word  # Track latest punctuation variant

        # Round-based tracking to require consecutive observations
        # - last_seen_round: the round id where this word was last observed
        # - seen_in_row: number of consecutive rounds the word has been observed
        # - last_seen_pos: the token index where the word was last observed (for position consistency)
        # - pos_consistent_in_row: number of consecutive rounds the word has been observed at approximately the same position
        self.last_seen_round = None if initial_round is None else initial_round
        self.seen_in_row = 1 if initial_round is not None else 0
        self.last_seen_pos = None
        self.pos_consistent_in_row = 0

    def increment_frequency(self, word_variant, current_round=None, current_pos=None):
        """Increment frequency for this word and update consecutive-round and position-consistency counters.

        Parameters
        ----------
        word_variant : str
            The original word form observed (may contain punctuation/capitalization).
        current_round : int|None
            The integer id of the current transcription round. If provided, updates
            the seen_in_row counter (consecutive rounds seen); otherwise only
            frequency and timestamps are updated.
        current_pos : int|None
            The index position within the current transcription where the word was observed.
        pos_tolerance : int
            How far (in token positions) an observation can drift and still count as position-consistent.
        """
        # Increment global frequency (capped externally by the tracker if desired)
        self.frequency += 1
        self.last_seen = time.time()

        # Update consecutive-round counters
        if current_round is not None:
            if self.last_seen_round is None:
                # First time we learn about the round
                self.seen_in_row = 1
            elif current_round == self.last_seen_round:
                # Same round observed multiple times (we count words once per round)
                # do not increment seen_in_row further
                pass
            elif current_round == self.last_seen_round + 1:
                # Observed in consecutive round
                self.seen_in_row += 1
            else:
                # Not consecutive - reset consecutive counter
                self.seen_in_row = 1
            self.last_seen_round = current_round

        # Update positional consistency
        if current_pos is not None:
            if self.last_seen_pos is None:
                self.pos_consistent_in_row = 1
            else:
                # If this observation is near the previous observed position, count as consistent
                if abs(current_pos - self.last_seen_pos) <= 1:
                    # If the round is consecutive (or same), increment pos_consistent_in_row
                    if current_round is None or self.last_seen_round is None or current_round == self.last_seen_round or current_round == self.last_seen_round + 1:
                        self.pos_consistent_in_row += 1
                    else:
                        # Position consistent but not consecutive round -> reset to 1
                        self.pos_consistent_in_row = 1
                else:
                    # Position drifted too far -> reset
                    self.pos_consistent_in_row = 1
            self.last_seen_pos = current_pos

    def get_key(self):
        """Return normalized word as key - context will be handled at tracker level"""
        return self.word_clean

    def should_graduate(self, min_frequency=4, min_consecutive=2, require_pos_consistency=False, min_pos_consistent=1):
        """Decide whether this word should be considered confirmed.

        We require BOTH a minimum total frequency and a minimum number of
        consecutive rounds observed to avoid intermittent words graduating.

        Optionally, require position-consistency (observed near the same token
        index across rounds) by setting require_pos_consistency=True.
        """
        freq_ok = self.frequency >= min_frequency
        consec_ok = self.seen_in_row >= min_consecutive

        # If a word has already been confirmed, be more lenient for subsequent appearances.
        # This helps repeated words (like 'do' in this case) to pass graduation
        # without being trapped by positional consistency checks.
        if self.state == "confirmed":
            return freq_ok and consec_ok

        if not require_pos_consistency:
            return freq_ok and consec_ok
        else:
            pos_ok = self.pos_consistent_in_row >= min_pos_consistent
            return freq_ok and consec_ok and pos_ok

    def get_output_word(self):
        """Return the word form to output (with latest punctuation)"""
        return self.latest_form

    def __str__(self):
        return f"{self.latest_form}({self.frequency}:{self.state}/r{self.seen_in_row})"


class TranscriptionTracker:
    def __init__(self, min_confirmed_words=4, min_frequency=4, min_consecutive_rounds=2):
        self.word_states = {}  # normalized_word -> WordState (back to simple keys for stability)
        self.confirmed_words = []  # Simple list of confirmed words
        self.sent_words_count = 0
        self.min_confirmed_words = min_confirmed_words  # Stop removing words when we reach this many
        self.last_start_index = 0  # Track last start_index to prevent large backtracks

        # Round tracking: each call to process_transcription increments the round_id
        # so we can track consecutive rounds a word appears in.
        self.current_round = 0

        # Confirmation thresholds
        self.min_frequency = min_frequency
        self.min_consecutive_rounds = min_consecutive_rounds

    def process_transcription(self, words, verbose=False):
        """Update internal state from a full transcription pass and return newly confirmed words.

        Instrumented with detailed alignment + graduation diagnostics when verbose=True.
        """
        new_words_to_send = []

        # Start a new round
        self.current_round += 1
        round_id = self.current_round

        # -----------------------------
        # 1. Update word observations
        # -----------------------------
        words_seen_this_round = set()
        for i, word in enumerate(words):
            norm = normalize_word_for_matching(word)
            if norm in words_seen_this_round:
                continue  # count each unique word once per round
            words_seen_this_round.add(norm)
            state = self.word_states.get(norm)
            if state is not None:
                if state.frequency < self.min_frequency:
                    state.increment_frequency(word, current_round=round_id, current_pos=i)
                else:
                    # At/above cap: still update recency + positional consistency
                    state.latest_form = word
                    if state.last_seen_round != round_id:
                        prev_round = state.last_seen_round
                        state.last_seen_round = round_id
                        if prev_round is None or round_id == prev_round + 1:
                            state.seen_in_row = state.seen_in_row + 1 if state.seen_in_row else 1
                        else:
                            state.seen_in_row = 1
                    if state.last_seen_pos is None:
                        state.last_seen_pos = i
                        state.pos_consistent_in_row = 1
                    else:
                        if abs(i - state.last_seen_pos) <= 1:
                            state.pos_consistent_in_row += 1
                        else:
                            state.pos_consistent_in_row = 1
                        state.last_seen_pos = i
            else:
                # New word
                state = WordState(word, frequency=1, initial_round=round_id)
                state.last_seen_pos = i
                state.pos_consistent_in_row = 1
                self.word_states[norm] = state

            # State graduation phases (unconfirmed -> potential) purely for debug visualization
            if state.state == "unconfirmed" and state.frequency >= max(2, self.min_frequency // 2):
                state.state = "potential"

        # -----------------------------
        # 2. Alignment (determine start_index)
        # -----------------------------
        start_index = 0
        words_normalized = [normalize_word_for_matching(w) for w in words]
        if self.confirmed_words:
            max_overlap = min(len(self.confirmed_words), len(words))
            if verbose:
                print(f"[ALIGN] Confirmed={len(self.confirmed_words)} Incoming={len(words)} MaxOverlap={max_overlap}")
            for overlap in range(max_overlap, 0, -1):
                tail = [normalize_word_for_matching(w) for w in self.confirmed_words[-overlap:]]
                head = words_normalized[:overlap]
                if verbose:
                    match_flag = '✓' if tail == head else '✗'
                    print(f"[ALIGN] OverlapTest k={overlap}: tail={' '.join(tail)} | head={' '.join(head)} -> {match_flag}")
                if tail == head:
                    start_index = overlap
                    if verbose:
                        print(f"[ALIGN] ExactPrefixMatch size={overlap} -> start_index={start_index}")
                    break

        min_allowed_start = len(self.confirmed_words)
        if start_index < min_allowed_start:
            found_partial_alignment = False
            best_forward_candidate = None  # (potential_start, lookback, i)
            best_backward_candidate = None  # closest (highest) potential_start < min_allowed_start
            if self.confirmed_words and words:
                max_lookback = min(5, len(self.confirmed_words))
                for lookback in range(max_lookback, 0, -1):
                    last_seq = [normalize_word_for_matching(w) for w in self.confirmed_words[-lookback:]]
                    if verbose:
                        print(f"[ALIGN] PartialSearch lookback={lookback} seq={' '.join(last_seq)}")
                    for i in range(len(words) - lookback + 1):
                        slice_seq = words_normalized[i:i+lookback]
                        if last_seq == slice_seq:
                            potential_start = i + lookback
                            max_backtrack = min(10, len(self.confirmed_words) // 2)
                            allowed_floor = len(self.confirmed_words) - max_backtrack
                            if verbose:
                                print(f"[ALIGN]  Candidate at i={i} -> potential_start={potential_start} allowed_floor={allowed_floor}")
                            # Track forward and backward separately; we will pick the safest after search
                            if potential_start >= min_allowed_start:
                                # Forward (or exact) continuation. Prefer the SMALLEST forward jump.
                                if best_forward_candidate is None or potential_start < best_forward_candidate[0]:
                                    best_forward_candidate = (potential_start, lookback, i)
                            else:
                                # Backward (some rollback). Prefer the LARGEST potential_start (minimal rollback) >= allowed_floor
                                if potential_start >= allowed_floor:
                                    if best_backward_candidate is None or potential_start > best_backward_candidate[0]:
                                        best_backward_candidate = (potential_start, lookback, i)
                # Decide which candidate to use
                chosen = None
                if best_forward_candidate is not None:
                    potential_start, lookback, i = best_forward_candidate
                    # Disallow forward jumps larger than +1 to prevent skipping unseen tokens
                    if potential_start > min_allowed_start + 1:
                        if verbose:
                            print(f"[ALIGN]  Forward candidate would skip tokens (potential_start={potential_start} > {min_allowed_start}+1). Clamping to {min_allowed_start}.")
                        start_index = min_allowed_start
                    else:
                        start_index = potential_start
                        chosen = ('forward', best_forward_candidate)
                elif best_backward_candidate is not None:
                    potential_start, lookback, i = best_backward_candidate
                    start_index = potential_start
                    chosen = ('backward', best_backward_candidate)
                if verbose:
                    if chosen:
                        direction, data = chosen
                        ps, lb, idx = data
                        print(f"[ALIGN]  Selected {direction} candidate start_index={start_index} (lookback={lb} at i={idx})")
                    else:
                        print(f"[ALIGN]  No suitable partial candidate found")
                found_partial_alignment = chosen is not None
            if not found_partial_alignment:
                if verbose:
                    print(f"[ALIGN] Fallback boundary correction {start_index}->{min_allowed_start}")
                start_index = min_allowed_start

        # Final safety: never allow forward skip beyond current confirmed length
        if start_index > len(self.confirmed_words):
            if verbose:
                print(f"[ALIGN] Safety clamp forward start_index {start_index}->{len(self.confirmed_words)}")
            start_index = len(self.confirmed_words)

        self.last_start_index = start_index

        # -----------------------------
        # 3. Graduation (monotonic sequence extension)
        # -----------------------------
        if verbose:
            print(f"[GRAD] Begin start_index={start_index} confirmed_len={len(self.confirmed_words)}")
        for i in range(start_index, len(words)):
            word = words[i]
            norm = normalize_word_for_matching(word)
            expected_index = len(self.confirmed_words)

            # Strict monotonic ordering: token index must equal expected_index exactly.
            if i != expected_index:
                if verbose:
                    reason = 'backtrack' if i < expected_index else 'forward drift'
                    print(f"[GRAD] Abort: token_index {i} != expected {expected_index} ({reason})")
                break

            state = self.word_states.get(norm)
            if state is None:
                if verbose:
                    print(f"[GRAD] Stop: '{word}' unseen state (norm={norm})")
                break

            # Sanity: ensure alignment didn't skip an expected next token from raw transcript
            if expected_index > 0:
                prev_confirmed_norm = normalize_word_for_matching(self.confirmed_words[-1])
                if i > 0:
                    prev_raw_norm = normalize_word_for_matching(words[i-1])
                    if prev_raw_norm != prev_confirmed_norm and verbose:
                        print(f"[GRAD] Warning: preceding raw token '{words[i-1]}' (norm={prev_raw_norm}) != last confirmed '{self.confirmed_words[-1]}' (norm={prev_confirmed_norm})")

            require_pos = True
            min_pos_consistent = 2
            freq_ok = state.frequency >= self.min_frequency
            consec_ok = state.seen_in_row >= self.min_consecutive_rounds
            pos_ok = state.pos_consistent_in_row >= min_pos_consistent
            can_graduate = freq_ok and consec_ok and (not require_pos or pos_ok)

            if verbose:
                print(f"[GRAD] Eval '{word}': idx={i} expected={expected_index} freq={state.frequency}/{self.min_frequency} consec={state.seen_in_row}/{self.min_consecutive_rounds} pos_cons={state.pos_consistent_in_row}/{min_pos_consistent} last_seen_pos={state.last_seen_pos} last_start_index={self.last_start_index} -> {'GRAD' if can_graduate else 'HOLD'}")

            if not can_graduate:
                break

            output_word = state.get_output_word()
            if self.confirmed_words and normalize_word_for_matching(self.confirmed_words[-1]) == norm:
                if verbose:
                    print(f"[GRAD] Skip duplicate '{output_word}' (same as last confirmed)")
                continue
            if state.last_seen_pos is not None and state.last_seen_pos < self.last_start_index:
                if verbose:
                    print(f"[GRAD] Abort: '{output_word}' last_seen_pos {state.last_seen_pos} < last_start_index {self.last_start_index}")
                break

            self.confirmed_words.append(output_word)
            state.state = "confirmed"
            new_words_to_send.append(output_word)
            self.sent_words_count += 1
            if verbose:
                print(f"[GRAD] ✅ Graduated '{output_word}' -> confirmed_len={len(self.confirmed_words)}")

        return new_words_to_send

    def _try_n_word_match(self, n, words, words_clean, search_start, fallback=False):  # Deprecated
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
            debug_str = f"{word_state.latest_form}({word_state.frequency})"
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




def simulate_streaming_transcription(audio_path: str, batch_words: list, chunk_duration_ms: int = 250, verbose: bool = False):
    """
    Simulate streaming transcription by processing audio in chunks like the WebSocket endpoint.
    Stops as soon as the streaming result diverges from the batch transcription.
    Returns (streaming_result, divergence_info).
    """
    asr_model = load_model()
    
    # Load and prepare audio
    audio = AudioSegment.from_file(audio_path)
    if audio.channels > 1:
        audio = audio.set_channels(1)
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    
    # Convert to raw bytes
    sample_rate = 16000
    sample_width = 2
    channels = 1
    window_size_seconds = 15
    min_audio_len_seconds = 3
    min_audio_len_bytes = min_audio_len_seconds * sample_rate * sample_width * channels
    window_size_bytes = window_size_seconds * sample_rate * sample_width * channels
    
    # Simulate streaming by processing chunks
    buffer = bytearray()
    tracker = TranscriptionTracker()
    streaming_result = []
    is_initial_transcription = True
    
    # Process audio in chunks
    total_length_ms = len(audio)
    chunk_size_ms = chunk_duration_ms
    divergence_info = None
    
    print(f"[STREAM-TEST] Simulating streaming with {chunk_size_ms}ms chunks")
    print(f"[STREAM-TEST] Total audio: {total_length_ms/1000:.2f}s")
    print(f"[STREAM-TEST] Target batch result: {' '.join(batch_words[:10])}{'...' if len(batch_words) > 10 else ''}")
    
    for chunk_start in range(0, total_length_ms, chunk_size_ms):
        chunk_end = min(chunk_start + chunk_size_ms, total_length_ms)
        chunk = audio[chunk_start:chunk_end]
        
        # Convert chunk to bytes and add to buffer
        chunk_bytes = chunk.raw_data
        buffer.extend(chunk_bytes)
        
        # Determine if we should transcribe
        should_transcribe = False
        if is_initial_transcription:
            if len(buffer) >= min_audio_len_bytes:
                should_transcribe = True
                is_initial_transcription = False
        else:
            should_transcribe = True
        
        if should_transcribe:
            # Convert buffer to AudioSegment for transcription
            audio_segment = AudioSegment(data=buffer, sample_width=sample_width, frame_rate=sample_rate, channels=channels)
            
            # Transcribe
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                temp_wav_file_path = tmpfile.name
                audio_segment.export(temp_wav_file_path, format="wav")
            
            try:
                # Suppress progress bars during transcription
                _suppress_progress_bars()
                transcription_result = asr_model.transcribe([temp_wav_file_path])
                _restore_stderr()
                current_words = transcription_result[0].text.strip().split() if transcription_result and transcription_result[0].text else []
                
                # Process through tracker with compact visual output
                new_words = tracker.process_transcription(current_words, verbose=False)
                if new_words:
                    streaming_result.extend(new_words)
                
                if verbose and current_words:
                    chunk_num = chunk_start//chunk_size_ms
                    # Create visual representation of word states
                    visual_words = []
                    confirmed_count = len(tracker.confirmed_words)

                    for i, word in enumerate(current_words):
                        word_key = normalize_word_for_matching(word)
                        is_align_point = (hasattr(tracker, 'last_start_index') and tracker.last_start_index > 0 and i == tracker.last_start_index)

                        if is_align_point:
                            # Alignment point - blue with |
                            visual_words.append(f"\033[94m|{word}\033[0m")
                        elif word_key in tracker.word_states:
                            state = tracker.word_states[word_key]
                            # Visualize positional consistency in the display
                            pos_ok = state.pos_consistent_in_row >= 2
                            if state.frequency >= tracker.min_frequency and pos_ok:
                                # Graduated word - green with []
                                visual_words.append(f"\033[92m[{word}]\033[0m")
                            elif state.frequency >= max(2, tracker.min_frequency // 2):
                                # Potential word - yellow with ()
                                visual_words.append(f"\033[93m({word})\033[0m")
                            else:
                                # Unconfirmed - red
                                visual_words.append(f"\033[91m{word}\033[0m")
                        else:
                            # New word - red
                            visual_words.append(f"\033[91m{word}\033[0m")

                    visual_text = ' '.join(visual_words)
                    new_count = len(new_words) if new_words else 0
                    print(f"[STREAM-TEST] Chunk {chunk_num}: +{new_count}→{len(streaming_result)} | {visual_text}")
                
                # Check for divergence after each chunk that produces words
                if streaming_result:
                    # Compare current streaming result with batch result using normalized matching
                    for i, stream_word in enumerate(streaming_result):
                        # Normalize both for robust comparison (strip punctuation, map number words)
                        stream_norm = normalize_word_for_matching(stream_word)
                        batch_norm = normalize_word_for_matching(batch_words[i]) if i < len(batch_words) else None

                        if i >= len(batch_words) or batch_norm is None or stream_norm != batch_norm:
                            # If it's merely capitalization or punctuation differences, the normalization will match above
                            # Also allow simple plural differences (e.g., word vs words)
                            is_plural_difference = False
                            if i < len(batch_words):
                                sw = stream_norm or ""
                                bw = batch_norm or ""
                                if sw + 's' == bw or bw + 's' == sw:
                                    is_plural_difference = True

                            if is_plural_difference:
                                continue
                            else:
                                # Real divergence - stop the test
                                divergence_info = {
                                    'position': i,
                                    'chunk_time': chunk_end / 1000.0,
                                    'streaming_words': len(streaming_result),
                                    'batch_words': len(batch_words),
                                    'stream_word': stream_word if i < len(streaming_result) else '[MISSING]',
                                    'batch_word': batch_words[i] if i < len(batch_words) else '[MISSING]',
                                    'streaming_result': streaming_result.copy(),
                                    'raw_transcription': current_words.copy(),
                                    'last_start_index': getattr(tracker, 'last_start_index', None),
                                    'tracker_debug': tracker.get_debug_info()
                                }
                                print(f"[STREAM-TEST] 🚨 DIVERGENCE DETECTED at position {i} after {chunk_end/1000:.2f}s")
                                print(f"[STREAM-TEST] Expected: '{batch_words[i] if i < len(batch_words) else '[MISSING]'}'")
                                print(f"[STREAM-TEST] Got: '{stream_word if i < len(streaming_result) else '[MISSING]'}'")
                                print(f"[STREAM-TEST] Tracker last_start_index={divergence_info['last_start_index']}")
                                print(f"[STREAM-TEST] Tracker debug: {divergence_info['tracker_debug']}")
                                return streaming_result, divergence_info
                
            finally:
                os.remove(temp_wav_file_path)
            
            # Manage buffer size
            if len(buffer) > window_size_bytes:
                buffer = buffer[-window_size_bytes:]
    
    # If we get here, no divergence was found
    return streaming_result, divergence_info


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
    """Loads the ASR model into the global ASR_MODEL variable if not already loaded.

    Heavy NeMo import occurs here to speed up initial module import.
    """
    global ASR_MODEL
    if ASR_MODEL is None:
        # Local import (heavy)
        import nemo.collections.asr as nemo_asr  # type: ignore
        check_ffmpeg_in_path()  # Check for ffmpeg before loading the model
        print("Loading Parakeet TDT model (lazy import)...")
        
        # Suppress progress bars during model loading
        _suppress_progress_bars()
        try:
            # Additional tqdm suppression for model loading
            import logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
            
            model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
            # Move to GPU if available
            try:
                ASR_MODEL = model.cuda()
            except Exception:
                ASR_MODEL = model  # Fallback to CPU
        finally:
            _restore_stderr()
        print("Model loaded.")
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
            _suppress_progress_bars()
            segment_transcription_result = asr_model.transcribe([temp_wav_file_path])
            _restore_stderr()

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
    min_audio_len_seconds = 3 # Min audio for initial transcription
    min_audio_len_bytes = min_audio_len_seconds * sample_rate * sample_width * channels
    window_size_bytes = window_size_seconds * sample_rate * sample_width * channels

    buffer = bytearray()
    tracker = TranscriptionTracker()
    is_initial_transcription = True
    last_processed_buffer_size = 0  # Track last processed buffer size to avoid duplicates

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
                # Skip if we're about to process the same buffer size again (duplicate processing)
                if len(buffer) == last_processed_buffer_size:
                    if verbose_logging:
                        print(f"[{time.strftime('%H:%M:%S')}] Timeout with same buffer size ({len(buffer)}), skipping duplicate")
                    continue
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
            # Suppress progress bars during transcription
            _suppress_progress_bars()
            try:
                transcription_result = asr_model.transcribe([temp_wav_file_path])
            finally:
                _restore_stderr()
            transcription_end_time = time.time()
            os.remove(temp_wav_file_path)

            current_words = transcription_result[0].text.strip().split() if transcription_result and transcription_result[0].text else []

            if verbose_logging:
                print(f"[{time.strftime('%H:%M:%S')}] Audio: {len(audio_segment)/1000:.2f}s, Transcribe time: {transcription_end_time-transcription_start_time:.2f}s, Raw: {' '.join(current_words)}")

            # Process through WordState tracker
            new_words_to_send = tracker.process_transcription(current_words, verbose_logging)
            
            # Update last processed buffer size
            last_processed_buffer_size = len(buffer) if buffer else 0

            if new_words_to_send:
                await websocket.send_text(" ".join(new_words_to_send))
                if verbose_logging:
                    print(f"→ Sent: {' '.join(new_words_to_send)}")
                    # Show cumulative sent words vs current raw transcription for comparison
                    print(f"→ Total sent so far: {' '.join(tracker.confirmed_words)}")
                    print(f"→ Current raw words: {' '.join(current_words)}")
                    debug_info = tracker.get_debug_info()
                    if debug_info['potential_words']:
                        print(f"→ Pending: {', '.join(debug_info['potential_words'][:3])}")
            elif verbose_logging:
                # Even when nothing is sent, show the comparison
                print(f"→ No new words sent")
                print(f"→ Total sent so far: {' '.join(tracker.confirmed_words)}")
                print(f"→ Current raw words: {' '.join(current_words)}")


            # Manage buffer: if it's longer than window_size_bytes, keep only the latest part
            # NEVER clear the buffer completely - it contains important context for alignment
            if len(buffer) > window_size_bytes:
                buffer = buffer[-window_size_bytes:]
                if verbose_logging:
                    print(f"[{time.strftime('%H:%M:%S')}] Buffer trimmed to {len(buffer)} bytes (window limit)")


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
    """Load (or return cached) speaker-diarization pipeline.

    Performs lazy imports for pyannote + torch + dotenv.
    """
    global DIARIZATION_PIPELINE
    if DIARIZATION_PIPELINE is None:
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except Exception:
            pass  # .env loading is optional
        hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")
        if hf_token == "":
            raise EnvironmentError(
                "HUGGINGFACE_ACCESS_TOKEN not set in environment or .env file; required for diarization pipeline."  # noqa: E501
            )
        import torch  # type: ignore
        from pyannote.audio import Pipeline  # type: ignore
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading pyannote speaker-diarization pipeline on {device} (lazy import)…")
        DIARIZATION_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        try:
            DIARIZATION_PIPELINE.to(device)
        except Exception:
            pass
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


# ---------------------------------------------------------------------------
# Speaker-embedding + clustering utilities
# ---------------------------------------------------------------------------

def load_embedding_model(model_type: str = "ecapa"):
    """Return a speaker-embedding model instance for *model_type* (cached).

    Lazy-loads heavy dependencies.
    """
    model_type = model_type.lower()
    if model_type in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_type]

    import torch  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Embeddings] Loading '{model_type}' model on {device} (lazy import)…")

    if model_type == "ecapa":
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding  # type: ignore
        model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)
    elif model_type == "titanet":
        import nemo.collections.asr as nemo_asr  # type: ignore
        model = (
            nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
            .to(device)
            .eval()
        )
    else:
        raise ValueError(f"Unsupported embedding model_type '{model_type}'")

    EMBEDDING_MODELS[model_type] = model
    return model


def compute_segment_embeddings(audio_path: str, segments, model_type: str = "ecapa"):
    """Compute and return list of embeddings for *segments*.

    model_type options:
      - "ecapa"
      - "titanet"
      - "combo"  → concatenation of ECAPA and Titanet vectors

    Heavy imports (pyannote, torch, soundfile) are done lazily.
    """

    model_type = model_type.lower()
    from pyannote.audio import Audio  # type: ignore
    from pyannote.core import Segment  # type: ignore
    import torch  # type: ignore
    import soundfile as sf  # type: ignore

    audio_helper = Audio(sample_rate=16000, mono="downmix")

    def _single_embedding(single_model: str, seg):
        mdl = load_embedding_model(single_model)
        if single_model == "titanet":
            waveform_t, _ = audio_helper.crop(audio_path, Segment(seg["start"], seg["end"]))
            waveform_np = waveform_t.squeeze().numpy()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
                export_path = tmpwav.name
            sf.write(export_path, waveform_np, 16000)
            emb_tensor = mdl.get_embedding(export_path)
            os.remove(export_path)
            emb_np = (
                emb_tensor.detach().cpu().numpy()
                if isinstance(emb_tensor, torch.Tensor)
                else emb_tensor
            ).squeeze()
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            return emb_np
        else:  # ecapa
            waveform, _ = audio_helper.crop(audio_path, Segment(seg["start"], seg["end"]))
            emb_out = mdl(waveform[None])
            emb_np = (
                emb_out.detach().cpu().numpy()
                if isinstance(emb_out, torch.Tensor)
                else emb_out
            ).squeeze()
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            return emb_np

    embeds = []
    for seg in segments:
        try:
            if model_type == "combo":
                ecapa_emb = _single_embedding("ecapa", seg)
                titan_emb = _single_embedding("titanet", seg)
                emb_np = np.concatenate([ecapa_emb, titan_emb])
                norm = np.linalg.norm(emb_np)
                if norm > 0:
                    emb_np = emb_np / norm
            else:
                emb_np = _single_embedding(model_type, seg)

            embeds.append(emb_np)
        except Exception as e:
            raise RuntimeError(f"Failed to compute embedding for segment {seg}: {e}")

    return embeds


def cluster_embeddings(embeddings):
    """Cluster embeddings with HDBSCAN and return label list.

    Lazily imports hdbscan.
    """
    if len(embeddings) <= 1:
        return [0] * len(embeddings)

    import hdbscan  # type: ignore

    emb_arr = np.vstack(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        metric="euclidean",
        min_samples=2,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(emb_arr)
    return labels.tolist()


def assign_cluster_labels(segments, labels):
    """Attach human-readable speaker_{n} label to each segment based on *labels*."""
    speaker_map = {}
    next_idx = 1
    for seg, lab in zip(segments, labels):
        if lab == -1:
            seg["speaker"] = "unknown"
        else:
            if lab not in speaker_map:
                speaker_map[lab] = f"speaker_{next_idx}"
                next_idx += 1
            seg["speaker"] = speaker_map[lab]
    return segments


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def save_embedding_projection_plot(embeddings, labels, out_path):
    """Project high-dim speaker embeddings to 2-D and save a scatter plot.

    Uses TSNE for <= 100 points; PCA otherwise (faster). Unknown labels (-1)
    are plotted in gray. Imports heavy libs lazily.
    """
    if len(embeddings) == 0:
        return None

    import matplotlib.pyplot as plt  # type: ignore
    from sklearn.manifold import TSNE  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore

    emb_arr = np.vstack(embeddings)

    if len(embeddings) <= 100:
        reducer = TSNE(n_components=2, perplexity=min(30, max(5, len(embeddings) // 2)), random_state=0)
        coords = reducer.fit_transform(emb_arr)
    else:
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(emb_arr)

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(6, 6))
    for lab in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lab]
        xs = coords[idxs, 0]
        ys = coords[idxs, 1]
        if lab == -1:
            color = "#888888"
            label_name = "unknown"
        else:
            color = cmap(lab % 10)
            label_name = f"cluster {lab}"
        plt.scatter(xs, ys, c=[color], label=label_name, alpha=0.7, edgecolors="none")

    plt.legend(loc="best", fontsize=8)
    plt.title("Speaker embedding clusters (2-D projection)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Post-processing: assign HDBSCAN noise (-1) points via k-nearest neighbors
# ---------------------------------------------------------------------------

def knn_fill_noise_labels(embeddings, labels, k: int = 5, distance_threshold: float = 2):
    """Assign cluster labels to HDBSCAN noise points using k-NN.

    Lazily imports sklearn.neighbors.
    """
    from sklearn.neighbors import NearestNeighbors  # type: ignore

    emb_arr = np.vstack(embeddings)
    labels = np.asarray(labels)

    core_mask = labels != -1
    noise_mask = labels == -1

    if core_mask.sum() == 0 or noise_mask.sum() == 0:
        return labels.tolist()

    core_emb = emb_arr[core_mask]
    core_labels = labels[core_mask]

    k = min(k, core_emb.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(core_emb)
    dists, idxs = nbrs.kneighbors(emb_arr[noise_mask])

    new_labels = labels.copy()
    for i_noise, (dist_row, idx_row) in enumerate(zip(dists, idxs)):
        mean_dist = dist_row.mean()
        if mean_dist <= distance_threshold:
            neighbor_labels = core_labels[idx_row]
            counts = np.bincount(neighbor_labels)
            pred_lab = counts.argmax()
            new_labels[np.where(noise_mask)[0][i_noise]] = pred_lab
    return new_labels.tolist()


# ---------------------------------------------------------------------------
# OpenAI name inference helper
# ---------------------------------------------------------------------------

def replace_speaker_ids_with_names(segments):
    """Call OpenAI API to infer names for speaker ids and apply replacements.

    Lazily imports dotenv + requests; only used when name_labels=True.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    import requests  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env file")

    dialogue_lines = [f"{seg['speaker']}: {seg['text']}" for seg in segments]
    dialogue_text = "\n".join(dialogue_lines)

    system_prompt = (
        "You are an assistant that extracts speaker NAMES explicitly mentioned in a conversation.\n"
        "Input format: each line starts with a speaker id like 'speaker_1:' followed by speech.\n"
        "If a speaker clearly says their own name or is addressed by name, map that id to the name.\n"
        "If NO spoken name can be found for an id, leave the id unchanged (do NOT invent roles or titles).\n"
        "Return ONLY a JSON object mapping original ids to new names.\n"
        "Names must be lower-case, words joined by single dashes (e.g. 'john-doe'). No spaces or other punctuation."
    )

    user_prompt = (
        "Here is the transcript:\n\n" + dialogue_text + "\n\n" +
        "Return the JSON mapping now."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-o4-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    try:
        mapping = json.loads(content)
    except json.JSONDecodeError:
        print("[OpenAI] Failed to parse JSON response, leaving labels unchanged.")
        return segments

    for seg in segments:
        if seg["speaker"] in mapping:
            seg["speaker"] = mapping[seg["speaker"]]
        if "segments" in seg:
            for sub in seg["segments"]:
                if sub["speaker"] in mapping:
                    sub["speaker"] = mapping[sub["speaker"]]

    return segments


# ---------------------------------------------------------------------------
# Endpoint: ASR + speaker embedding clustering (no full diarization)
# ---------------------------------------------------------------------------

@app.post("/transcribe_cluster")
async def transcribe_cluster_endpoint(
    audio_file: UploadFile = File(...),
    model: str = Query("ecapa", description="Speaker embedding model: 'ecapa' or 'titanet"),
    name_labels: bool = Query(False, description="Use OpenAI to guess names and replace speaker ids"),
):
    """Transcribe audio and cluster segments by speaker embeddings.

    Returns a JSON with merged segments (same format as `/transcribe_diarize`)
    but speaker labels are derived from HDBSCAN clustering of ECAPA embeddings.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save upload
            temp_input_path = os.path.join(temp_dir, audio_file.filename)
            with open(temp_input_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            # Extract audio if video & ensure mono-16k WAV
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
                asr_audio_path = extract_audio_from_video(temp_input_path, temp_dir)
            else:
                asr_audio_path = temp_input_path

            asr_audio_path = convert_audio_to_wav_mono_16k(asr_audio_path, temp_dir)

            # ASR with timestamps
            segment_length_sec = getattr(app.state, "segment_length_sec", 60)
            transcription, asr_segments = process_and_transcribe_audio_file_with_timestamps(
                asr_audio_path, segment_length_sec
            )

            if transcription.startswith("Error:"):
                return JSONResponse(
                    {"error": "ASR processing failed", "details": transcription},
                    status_code=500,
                )

            # Speaker embeddings & clustering
            embeddings = compute_segment_embeddings(asr_audio_path, asr_segments, model_type=model.lower())
            labels = cluster_embeddings(embeddings)
            # Fill HDBSCAN noise points using k-NN to nearest clusters
            labels = knn_fill_noise_labels(embeddings, labels)
            annotated_segments = assign_cluster_labels(asr_segments, labels)

            # Optionally merge consecutive identical speakers for compactness
            joined_segments = join_consecutive_segments_by_speaker(annotated_segments)

            # ------------------------------------------------------------------
            # Optional: replace speaker IDs with human-friendly names via OpenAI
            # ------------------------------------------------------------------
            if name_labels:
                try:
                    joined_segments = replace_speaker_ids_with_names(joined_segments)
                except Exception as e:
                    print(f"Name replacement failed: {e}")

            # Save embedding projection plot for debugging/inspection
            try:
                plot_filename = f"embedding_projection_{uuid.uuid4().hex[:8]}.png"
                print(f"Saving embedding plot to {plot_filename}")
                plot_path = os.path.join("/tmp", plot_filename)
                save_embedding_projection_plot(embeddings, labels, plot_path)
            except Exception as e:
                print(f"Failed to generate embedding plot: {e}")
                plot_path = None

            response_payload = {"segments": joined_segments}
            if plot_path:
                response_payload["embedding_plot"] = plot_path

            return JSONResponse(response_payload)

    except Exception as e:
        print(f"Error in /transcribe_cluster endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    parser.add_argument("--stream-test", action="store_true",
                        help="Test streaming vs batch transcription on an audio file and compare outputs.")

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

        if args.stream_test:
            print(f"Running stream test comparison for: {args.audio_file}")
            print("=" * 80)
            
            # Run batch transcription first
            print("1. BATCH TRANSCRIPTION:")
            print("-" * 40)
            batch_result = process_and_transcribe_audio_file(args.audio_file, args.segment_length)
            if batch_result.startswith("Error:"):
                print(f"Batch transcription failed: {batch_result}")
                exit(1)
            
            batch_words = batch_result.strip().split()
            print(f"Batch result: {len(batch_words)} words")
            
            print("\n2. STREAMING TRANSCRIPTION (until divergence):")
            print("-" * 40)
            print("Legend: \033[92m[word]\033[0m=confirmed(5+) \033[93m(word)\033[0m=potential(2-4) \033[91mword\033[0m=new(1) \033[94m|word\033[0m=align-point")
            print("-" * 40)
            streaming_result, divergence_info = simulate_streaming_transcription(args.audio_file, batch_words, verbose=args.verbose)
            
            print("\n3. ANALYSIS:")
            print("-" * 40)
            
            if divergence_info:
                print(f"🚨 DIVERGENCE FOUND:")
                print(f"  • Position: {divergence_info['position']}")
                print(f"  • Time: {divergence_info['chunk_time']:.2f}s into audio")
                print(f"  • Expected word: '{divergence_info['batch_word']}'")
                print(f"  • Streaming word: '{divergence_info['stream_word']}'")
                print(f"  • Streaming words so far: {divergence_info['streaming_words']}")
                print(f"  • Raw transcription at divergence: {' '.join(divergence_info['raw_transcription'])}")
                
                # Show context around divergence
                pos = divergence_info['position']
                context_start = max(0, pos - 3)
                context_end = min(len(batch_words), pos + 4)
                
                print(f"\n  📍 CONTEXT AROUND DIVERGENCE:")
                print(f"     Batch:     {' '.join(batch_words[context_start:context_end])}")
                print(f"     Streaming: {' '.join(streaming_result[context_start:context_end] if context_end <= len(streaming_result) else streaming_result[context_start:] + ['[MISSING]'] * (context_end - len(streaming_result)))}")
                
                # Calculate partial similarity up to divergence
                matching_words = divergence_info['position']
                total_words = max(len(batch_words), len(streaming_result))
                similarity = matching_words / total_words * 100 if total_words > 0 else 0
                print(f"\n  📊 SIMILARITY: {similarity:.1f}% ({matching_words}/{total_words} words match before divergence)")
                
            else:
                streaming_words = streaming_result if isinstance(streaming_result, list) else streaming_result.split()
                if len(streaming_words) == len(batch_words) and all(s == b for s, b in zip(streaming_words, batch_words)):
                    print("✅ PERFECT MATCH! Streaming and batch results are identical.")
                else:
                    print("⚠️  No early divergence found, but results may differ at the end.")
                    print(f"   Streaming: {len(streaming_words)} words")
                    print(f"   Batch: {len(batch_words)} words")
            
        else:
            print(f"Running in CLI mode for input file: {args.audio_file}")
            # process_and_transcribe_audio_file calls load_model() internally.
            final_transcription = process_and_transcribe_audio_file(args.audio_file, args.segment_length)

            if final_transcription.startswith("Error:") or "[Transcription Failed for Segment]" in final_transcription:
                print(f"\nCLI Transcription Failed/Error: {final_transcription}")
            else:
                print("\nFull Transcription:")
                print(final_transcription)

            print("CLI transcription process complete.")
