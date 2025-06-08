import tempfile
import time
import os
import asyncio
import re
from collections import deque
from fastapi import WebSocket, WebSocketDisconnect
from pydub import AudioSegment


def strip_punctuation(word):
    """Remove punctuation and normalize capitalization for matching purposes"""
    return re.sub(r'[^\w]', '', word).lower()


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
    match = re.match(r'([£$€¥₹]?\d+)', word.lower())
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


def sequence_matches_flexible(target_seq, words, start_pos):
    """Check if target sequence matches at start_pos with flexible number matching and currency conversion"""
    # Handle currency sequences like "three dollars" -> "$3"
    converted_words = []
    i = 0
    while i < len(words):
        if i < len(words) - 1:
            current = normalize_word_for_matching(words[i])
            next_word = normalize_word_for_matching(words[i + 1])
            
            # Check for currency sequence
            if current.isdigit() and next_word in ['dollar', 'dollars', 'pound', 'pounds', 'euro', 'euros']:
                if next_word in ['dollar', 'dollars']:
                    converted_words.append(f"${current}")
                elif next_word in ['pound', 'pounds']:
                    converted_words.append(f"£{current}")
                elif next_word in ['euro', 'euros']:
                    converted_words.append(f"€{current}")
                i += 2  # Skip the currency word
                continue
        
        converted_words.append(normalize_word_for_matching(words[i]))
        i += 1
    
    # Now check the sequence against converted words
    if start_pos + len(target_seq) > len(converted_words):
        return False
    
    for i, target_word in enumerate(target_seq):
        current_word = converted_words[start_pos + i]
        target_norm = normalize_word_for_matching(target_word)
        
        # Also try direct flexible matching for number variations
        if current_word != target_norm and not words_match_flexible(target_word, current_word):
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
        if self.frequency >= 3:
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
    def __init__(self):
        self.word_states = {}  # context_key -> WordState
        self.confirmed_words = []  # Simple list of confirmed words (no context)
        self.sent_words_count = 0
        
    def process_transcription(self, words):
        new_words_to_send = []
        
        # Update word frequencies using context-aware keys
        for i, word in enumerate(words):
            prev_word = words[i-1] if i > 0 else None
            next_word = words[i+1] if i < len(words)-1 else None
            
            word_state = WordState(word, prev_word, next_word)
            context_key = word_state.get_context_key()
            
            if context_key in self.word_states:
                # Only increment if not already confirmed (to stop counting at 3)
                existing_state = self.word_states[context_key]
                if existing_state.frequency < 3:
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
        
        # Try cascading fallback: 4 words -> 3 words -> 2 words -> 1 word
        if len(self.confirmed_words) >= 4:
            # Try last four words first
            last_four = [strip_punctuation(w) for w in self.confirmed_words[-4:]]
            # Debug logging
            print(f"[DEBUG] Looking for last 4: {last_four} in last 20 words: {words_clean[search_start:]}")
            for i in range(search_start, len(words_clean) - 3):
                if sequence_matches_flexible(last_four, words, i):
                    start_index = i + 4
                    print(f"[DEBUG] Found 4-word match at position {i}, start_index = {start_index}")
                    break
        
        # If no match with 4, try last three
        if start_index == 0 and len(self.confirmed_words) >= 3:
            last_three = [strip_punctuation(w) for w in self.confirmed_words[-3:]]
            # Debug logging
            print(f"[DEBUG] Looking for last 3: {last_three} in last 20 words: {words_clean[search_start:]}")
            for i in range(search_start, len(words_clean) - 2):
                if sequence_matches_flexible(last_three, words, i):
                    start_index = i + 3
                    print(f"[DEBUG] Found 3-word match at position {i}, start_index = {start_index}")
                    break
            
            # If no match with 3, try last two
            if start_index == 0:
                last_two = [strip_punctuation(w) for w in self.confirmed_words[-2:]]
                print(f"[DEBUG] Looking for last 2: {last_two} in last 20 words: {words_clean[search_start:]}")
                for i in range(search_start, len(words_clean) - 1):
                    if sequence_matches_flexible(last_two, words, i):
                        start_index = i + 2
                        print(f"[DEBUG] Found 2-word match at position {i}, start_index = {start_index}")
                        break
        
        # If still no match, or we have 1-2 confirmed words, try last two/one
        if start_index == 0 and len(self.confirmed_words) >= 2:
            last_two = [strip_punctuation(w) for w in self.confirmed_words[-2:]]
            print(f"[DEBUG] Looking for last 2 (fallback): {last_two} in last 20 words: {words_clean[search_start:]}")
            for i in range(search_start, len(words_clean) - 1):
                if sequence_matches_flexible(last_two, words, i):
                    start_index = i + 2
                    print(f"[DEBUG] Found 2-word fallback match at position {i}, start_index = {start_index}")
                    break
        
        # Final fallback: try just the last word
        if start_index == 0 and len(self.confirmed_words) >= 1:
            last_word = strip_punctuation(self.confirmed_words[-1])
            print(f"[DEBUG] Looking for last 1: [{last_word}] in last 20 words: {words_clean[search_start:]}")
            for i in range(search_start, len(words_clean)):
                if words_match_flexible(last_word, words[i]):
                    start_index = i + 1
                    print(f"[DEBUG] Found 1-word match at position {i}, start_index = {start_index}")
                    break
        
        # If still no match found, throw exception to stop program
        if start_index == 0 and len(self.confirmed_words) > 0:
            raise Exception(f"Sequence matching failed completely! Confirmed words: {self.confirmed_words[-5:]} | Current transcription: {words[:20]}")
        
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
    
    def get_debug_info(self):
        confirmed_words = []
        potential_words = []
        
        for word_state in self.word_states.values():
            context_str = f"[{word_state.prev_word or ''}-{word_state.next_word or ''}]"
            if word_state.state == "confirmed":
                confirmed_words.append(f"{word_state.latest_form}({word_state.frequency}){context_str}")
            elif word_state.state == "potential":
                potential_words.append(f"{word_state.latest_form}({word_state.frequency}){context_str}")
        
        # Sort by frequency (descending) and take last 5
        confirmed_words.sort(key=lambda x: int(x.split('(')[1].split(')')[0]), reverse=True)
        potential_words.sort(key=lambda x: int(x.split('(')[1].split(')')[0]), reverse=True)
        
        return {
            "last_5_confirmed": confirmed_words[:5],
            "potential_words": potential_words[:5]
        }


async def websocket_transcribe_v3_endpoint(websocket: WebSocket, asr_model, verbose=False):
    await websocket.accept()
    
    # Get chunk duration from query params
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
    tracker = TranscriptionTracker()
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
                
                silence_duration_ms = max(0, (min_audio_len_bytes - len(buffer)) * 1000 // (sample_rate * sample_width))
                if silence_duration_ms > 0:
                    padding = AudioSegment.silent(duration=silence_duration_ms, frame_rate=sample_rate)
                    audio_data = padding + AudioSegment(data=buffer, sample_width=sample_width, frame_rate=sample_rate, channels=channels)
                else:
                    audio_data = AudioSegment(data=buffer, sample_width=sample_width, frame_rate=sample_rate, channels=channels)
            else:
                if len(buffer) > window_size_bytes:
                    buffer = buffer[-window_size_bytes:]
                audio_data = AudioSegment(data=buffer, sample_width=sample_width, frame_rate=sample_rate, channels=channels)
            
            # Transcribe audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                temp_wav_file_path = tmpfile.name
                audio_data.export(temp_wav_file_path, format="wav")
            
            start_time = time.time()
            transcription_result = asr_model.transcribe([temp_wav_file_path])
            end_time = time.time()
            os.remove(temp_wav_file_path)
            
            current_words = transcription_result[0].text.strip().split() if transcription_result and transcription_result[0].text else []
            
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] Audio: {len(audio_data)/1000:.2f}s, Transcribe time: {end_time-start_time:.2f}s, Raw: {' '.join(current_words)}")
            
            # Process through WordState tracker
            new_words = tracker.process_transcription(current_words)
            
            if new_words:
                await websocket.send_text(" ".join(new_words))
                if verbose:
                    last_5_confirmed = tracker.confirmed_words[-5:] if len(tracker.confirmed_words) >= 5 else tracker.confirmed_words
                    debug_info = tracker.get_debug_info()
                    print(f"Sent: {' '.join(new_words)} | Last 5 Confirmed: {last_5_confirmed} | Potential: {debug_info['potential_words']}")
    
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket V3.")
    except Exception as e:
        print(f"An error occurred in WebSocket V3: {e}")
    finally:
        if websocket.client_state.value != 3:
            await websocket.close()