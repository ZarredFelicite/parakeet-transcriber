import re
import time

def strip_punctuation(word):
    return re.sub(r'[^\w\s]', '', word).lower()


def word_to_number(word):
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
    clean = strip_punctuation(word)
    converted = word_to_number(clean)
    return converted

class WordState:
    def __init__(self, word, frequency=1, initial_round=None):
        self.word = word
        self.word_clean = normalize_word_for_matching(word)
        self.frequency = frequency
        self.state = "unconfirmed"
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.latest_form = word
        self.last_seen_round = None if initial_round is None else initial_round
        self.seen_in_row = 1 if initial_round is not None else 0
        self.last_seen_pos = None
        self.pos_consistent_in_row = 0

    def increment_frequency(self, word_variant, current_round=None, current_pos=None, pos_tolerance=1):
        self.frequency += 1
        self.last_seen = time.time()
        if current_round is not None:
            if self.last_seen_round is None:
                self.seen_in_row = 1
            elif current_round == self.last_seen_round:
                pass
            elif current_round == self.last_seen_round + 1:
                self.seen_in_row += 1
            else:
                self.seen_in_row = 1
            self.last_seen_round = current_round
        if current_pos is not None:
            if self.last_seen_pos is None:
                self.pos_consistent_in_row = 1
            else:
                if abs(current_pos - self.last_seen_pos) <= pos_tolerance:
                    if current_round is None or self.last_seen_round is None or current_round == self.last_seen_round or current_round == self.last_seen_round + 1:
                        self.pos_consistent_in_row += 1
                    else:
                        self.pos_consistent_in_row = 1
                else:
                    self.pos_consistent_in_row = 1
            self.last_seen_pos = current_pos

    def get_key(self):
        return self.word_clean

    def should_graduate(self, min_frequency=4, min_consecutive=2, require_pos_consistency=False, min_pos_consistent=1, confirmed_words=None):
        freq_ok = self.frequency >= min_frequency
        consec_ok = self.seen_in_row >= min_consecutive
        if not require_pos_consistency:
            return freq_ok and consec_ok
        else:
            pos_ok = self.pos_consistent_in_row >= min_pos_consistent
            
            # Special handling for repeated words: if this word has already been
            # confirmed before, relax the positional consistency requirement
            if not pos_ok and confirmed_words is not None:
                word_already_confirmed = any(
                    normalize_word_for_matching(confirmed_word) == self.word_clean 
                    for confirmed_word in confirmed_words
                )
                if word_already_confirmed and freq_ok and consec_ok:
                    # For repeated words, only require basic frequency and consecutive observation
                    return True
            
            return freq_ok and consec_ok and pos_ok

    def get_output_word(self):
        return self.latest_form

    def __str__(self):
        return f"{self.latest_form}({self.frequency}:{self.state}/r{self.seen_in_row})"

class TranscriptionTracker:
    def __init__(self, min_confirmed_words=4, min_frequency=4, min_consecutive_rounds=2):
        self.word_states = {}
        self.confirmed_words = []
        self.sent_words_count = 0
        self.min_confirmed_words = min_confirmed_words
        self.last_start_index = 0
        self.stall_count = 0  # Track how many rounds we've been stuck at the same position
        self.stall_threshold = 10  # Force more aggressive alignment after this many stalls
        self.current_round = 0
        self.min_frequency = min_frequency
        self.min_consecutive_rounds = min_consecutive_rounds
        
        # Context-aware tracking
        self.context_window = 2  # How many words of context to use
        self.use_context_aware = False  # Temporarily disable context-aware tracking for debugging

    def _get_context(self, words, position):
        """Get context for a word at given position."""
        if not self.use_context_aware:
            return None
            
        context = []
        
        # Add previous words (up to context_window)
        for i in range(max(0, position - self.context_window), position):
            context.append(words[i] if i < len(words) else None)
        
        # Add the word itself
        context.append(words[position] if position < len(words) else None)
        
        # Add next words (up to 1 for now)
        for i in range(position + 1, min(len(words), position + 2)):
            context.append(words[i] if i < len(words) else None)
        
        # Pad with None if needed to maintain consistent context size
        while len(context) < self.context_window + 2:  # prev + word + next
            context.append(None)
        
        return context

    def process_transcription(self, words, verbose=False):
        new_words_to_send = []
        self.current_round += 1
        words_seen_this_round = set()
        for i, word in enumerate(words):
            # Get context for this word
            context = self._get_context(words, i)
            
            # Create context-aware key with fuzzy matching
            if self.use_context_aware and context is not None:
                # Use only the core context (word + immediate neighbors) for more stability
                core_context = context[1:4] if len(context) >= 4 else context  # prev1, word, next1
                word_key = f"{normalize_word_for_matching(word)}@{'-'.join([normalize_word_for_matching(w) if w else '_' for w in core_context])}"
            else:
                word_key = normalize_word_for_matching(word)
            
            if word_key in words_seen_this_round:
                continue  # count each unique contextual word once per round
            words_seen_this_round.add(word_key)
            if word_key in self.word_states:
                existing_state = self.word_states[word_key]
                if existing_state.frequency < self.min_frequency:
                    existing_state.increment_frequency(word, current_round=self.current_round, current_pos=i)
                else:
                    existing_state.latest_form = word
                    if existing_state.last_seen_round != self.current_round:
                        existing_state.last_seen_round = self.current_round
                        existing_state.seen_in_row = existing_state.seen_in_row + 1 if existing_state.seen_in_row else 1
                    if existing_state.last_seen_pos is None:
                        existing_state.last_seen_pos = i
                        existing_state.pos_consistent_in_row = 1
                    else:
                        if abs(i - existing_state.last_seen_pos) <= 1:
                            existing_state.pos_consistent_in_row += 1
                        else:
                            existing_state.pos_consistent_in_row = 1
                        existing_state.last_seen_pos = i
            else:
                word_state = WordState(word, frequency=1, initial_round=self.current_round)
                word_state.last_seen_pos = i
                word_state.pos_consistent_in_row = 1
                self.word_states[word_key] = word_state

        # Find start_index
        start_index = 0
        words_normalized = [normalize_word_for_matching(w) for w in words]
        if self.confirmed_words:
            max_overlap = min(len(self.confirmed_words), len(words))
            for overlap in range(max_overlap, 0, -1):
                tail = [normalize_word_for_matching(w) for w in self.confirmed_words[-overlap:]]
                head = words_normalized[:overlap]
                if tail == head:
                    start_index = overlap
                    break
        min_allowed_start = len(self.confirmed_words)
        found_partial_alignment = False
        if start_index < min_allowed_start:
            if self.confirmed_words and len(words) > 0:
                for lookback in range(min(5, len(self.confirmed_words)), 0, -1):
                    last_words = [normalize_word_for_matching(w) for w in self.confirmed_words[-lookback:]]
                    for i in range(len(words) - lookback + 1):
                        current_slice = words_normalized[i:i+lookback]
                        if last_words == current_slice:
                            potential_start = i + lookback
                            max_backtrack = min(10, len(self.confirmed_words) // 2)
                            if potential_start >= len(self.confirmed_words) - max_backtrack:
                                start_index = potential_start
                                found_partial_alignment = True
                                break
                    if found_partial_alignment:
                        break
            if not found_partial_alignment:
                start_index = min_allowed_start
        
        # Track alignment stalls FIRST
        if start_index == self.last_start_index:
            self.stall_count += 1
        else:
            self.stall_count = 0
        
        # Check if we're in a stall situation and need to force progress
        if self.stall_count >= self.stall_threshold and not found_partial_alignment:
            # Force more aggressive alignment to break the stall
            if verbose:
                print(f"[ALIGN] Stall detected ({self.stall_count} rounds), forcing alignment to {len(self.confirmed_words)}")
            start_index = len(self.confirmed_words)
            self.stall_count = 0  # Reset stall counter after forcing alignment
        elif not found_partial_alignment and start_index < len(self.confirmed_words):
            # Conservative fallback: only advance by 1 position to avoid skipping words
            conservative_start = max(start_index, self.last_start_index + 1)
            if conservative_start > len(self.confirmed_words):
                conservative_start = len(self.confirmed_words)
            if verbose:
                print(f"[ALIGN] Conservative fallback {start_index}->{conservative_start} (stall_count={self.stall_count})")
            start_index = conservative_start
        
        self.last_start_index = start_index

        # Graduation with monotonic guard
        if verbose:
            print(f"[GRAD] Starting from index {start_index}, confirmed_words: {len(self.confirmed_words)}")
        for i in range(start_index, len(words)):
            word = words[i]
            
            # Get context for this word in graduation phase
            context = self._get_context(words, i)
            
            # Create context-aware key for graduation with fuzzy matching
            if self.use_context_aware and context is not None:
                # Use only the core context (word + immediate neighbors) for more stability
                core_context = context[1:4] if len(context) >= 4 else context  # prev1, word, next1
                word_key = f"{normalize_word_for_matching(word)}@{'-'.join([normalize_word_for_matching(w) if w else '_' for w in core_context])}"
            else:
                word_key = normalize_word_for_matching(word)
            expected_index = len(self.confirmed_words)
            # Allow small position drifts due to ASR transcript instability
            position_tolerance = 1
            if abs(i - expected_index) > position_tolerance:
                if verbose:
                    reason = 'backtrack' if i < expected_index else 'forward drift'
                    print(f"[GRAD] Position mismatch: token index {i} != expected confirmed index {expected_index} ({reason}). Aborting graduation to preserve order.")
                break
            elif i != expected_index:
                if verbose:
                    drift_direction = 'behind' if i < expected_index else 'ahead'
                    print(f"[GRAD] Small position drift: token index {i} vs expected {expected_index} ({drift_direction}). Allowing with tolerance.")
                # Adjust expected_index to match current position for this iteration
                expected_index = i
            if word_key in self.word_states:
                word_state = self.word_states[word_key]
                require_pos = True
                min_pos_consistent = 2
                
                # Use relaxed thresholds for context-aware tracking
                if self.use_context_aware and "@" in word_key:
                    # Context-aware words need less consecutive rounds since contexts are more specific
                    min_consecutive = max(1, self.min_consecutive_rounds - 1)
                    min_frequency = max(1, self.min_frequency - 1)
                else:
                    min_consecutive = self.min_consecutive_rounds
                    min_frequency = self.min_frequency
                
                if word_state.should_graduate(min_frequency=min_frequency, min_consecutive=min_consecutive, require_pos_consistency=require_pos, min_pos_consistent=min_pos_consistent, confirmed_words=self.confirmed_words):
                    output_word = word_state.get_output_word()
                    if (self.confirmed_words and normalize_word_for_matching(self.confirmed_words[-1]) == normalize_word_for_matching(output_word)):
                        if verbose:
                            print(f"[GRAD] Skipped duplicate '{output_word}' (same as last confirmed word)")
                        continue
                    # Check if this word is being seen at a position behind where we expect it
                    # But allow repeated words to graduate at their new position
                    if word_state.last_seen_pos is not None and word_state.last_seen_pos < expected_index:
                        # Check if this is a repeated word that appears later in the transcript
                        word_already_confirmed = any(
                            normalize_word_for_matching(confirmed_word) == word_key 
                            for confirmed_word in self.confirmed_words
                        )
                        if word_already_confirmed:
                            # This is a repeated word appearing later - allow it to graduate
                            if verbose:
                                print(f"[GRAD] Allowing repeated word '{output_word}' to graduate at position {expected_index} (previously at {word_state.last_seen_pos})")
                        else:
                            # This is a word appearing out of order - skip it
                            if verbose:
                                print(f"[GRAD] Skipped '{output_word}' because its last_seen_pos={word_state.last_seen_pos} < expected_index={expected_index}")
                            break
                    self.confirmed_words.append(output_word)
                    new_words_to_send.append(output_word)
                    self.sent_words_count += 1
                    if verbose:
                        print(f"[GRAD] Graduated '{output_word}' (freq={word_state.frequency}/{self.min_frequency}, consec={word_state.seen_in_row}/{self.min_consecutive_rounds}, pos_consistent={word_state.pos_consistent_in_row}/{min_pos_consistent})")
                else:
                    if verbose:
                        print(f"[GRAD] Stopped at '{word}' (freq={word_state.frequency}/{self.min_frequency}, consec={word_state.seen_in_row}/{self.min_consecutive_rounds}, pos_consistent={word_state.pos_consistent_in_row})")
                    break
            else:
                if verbose:
                    print(f"[GRAD] Stopped at '{word}' - not seen before")
                break
        return new_words_to_send

# --- Test ---
if __name__ == '__main__':
    print('Normalization test:', normalize_word_for_matching('with.'), normalize_word_for_matching('with'))
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    rounds = [
        ['hello','this','is','a','test'],
        ['hello','obviously','this','is','a','test'],
        ['hello','this','is','a','test'],
        ['hello','this','is','a','test']
    ]
    for r_idx, words in enumerate(rounds, start=1):
        new = tracker.process_transcription(words, verbose=True)
        print(f'After round {r_idx}, new_words_to_send: {new}')
        print(f'Confirmed words: {tracker.confirmed_words}\n')
