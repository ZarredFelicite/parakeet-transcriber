#!/usr/bin/env python3
"""
Context-aware word tracking for ASR transcription.

Instead of tracking words in isolation, this tracks words within their context
(e.g., [because, like, obviously] vs [that, like, you're]).
"""

import time
from typing import List, Optional, Tuple, Dict, Any

def strip_punctuation(word):
    """Remove punctuation from word for normalization."""
    return ''.join(c for c in word if c.isalnum())

def word_to_number(word):
    """Convert word numbers to digits for normalization."""
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12'
    }
    return word_to_num.get(word.lower(), word)

def normalize_word_for_matching(word):
    """Normalize word for consistent matching."""
    if not word:
        return ""
    normalized = strip_punctuation(word).lower()
    normalized = word_to_number(normalized)
    return normalized

class ContextualWordState:
    """Track a word within its specific context (surrounding words)."""
    
    def __init__(self, word: str, context: List[str], frequency: int = 1, initial_round: Optional[int] = None):
        self.word = word  # Original word with punctuation
        self.word_clean = normalize_word_for_matching(word)  # Normalized for robust matching
        self.context = context  # List of surrounding words (normalized)
        self.context_key = self._create_context_key(context)  # Unique key for this context
        self.frequency = frequency
        self.state = "unconfirmed"  # "unconfirmed", "potential", "confirmed"
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.latest_form = word  # Track latest punctuation variant
        
        # Round-based tracking
        self.last_seen_round = None if initial_round is None else initial_round
        self.seen_in_row = 1 if initial_round is not None else 0
        self.last_seen_pos = None
        self.pos_consistent_in_row = 0
    
    def _create_context_key(self, context: List[str]) -> str:
        """Create a unique key for this context."""
        # Use the last 2 words before and first 1 word after as context
        # Format: "prev2|prev1|WORD|next1"
        context_parts = []
        for word in context:
            if word is None:
                context_parts.append("_")
            else:
                context_parts.append(normalize_word_for_matching(word))
        return "|".join(context_parts)
    
    def increment_frequency(self, word_variant: str, current_round: Optional[int] = None, 
                          current_pos: Optional[int] = None, pos_tolerance: int = 1):
        """Increment frequency for this contextual word instance."""
        self.frequency += 1
        self.last_seen = time.time()
        self.latest_form = word_variant
        
        # Update consecutive-round counters
        if current_round is not None:
            if self.last_seen_round is None:
                self.seen_in_row = 1
            elif current_round == self.last_seen_round:
                # Same round observed multiple times - don't increment
                pass
            elif current_round == self.last_seen_round + 1:
                # Consecutive round
                self.seen_in_row += 1
            else:
                # Not consecutive - reset
                self.seen_in_row = 1
            self.last_seen_round = current_round
        
        # Update positional consistency
        if current_pos is not None:
            if self.last_seen_pos is None:
                self.pos_consistent_in_row = 1
            else:
                if abs(current_pos - self.last_seen_pos) <= pos_tolerance:
                    if (current_round is None or self.last_seen_round is None or 
                        current_round == self.last_seen_round or 
                        current_round == self.last_seen_round + 1):
                        self.pos_consistent_in_row += 1
                    else:
                        self.pos_consistent_in_row = 1
                else:
                    self.pos_consistent_in_row = 1
            self.last_seen_pos = current_pos
    
    def should_graduate(self, min_frequency: int = 4, min_consecutive: int = 2, 
                       require_pos_consistency: bool = False, min_pos_consistent: int = 1) -> bool:
        """Decide whether this contextual word should be confirmed."""
        freq_ok = self.frequency >= min_frequency
        consec_ok = self.seen_in_row >= min_consecutive
        
        if not require_pos_consistency:
            return freq_ok and consec_ok
        else:
            pos_ok = self.pos_consistent_in_row >= min_pos_consistent
            return freq_ok and consec_ok and pos_ok
    
    def get_output_word(self) -> str:
        """Return the word form to output."""
        return self.latest_form
    
    def __str__(self):
        return f"{self.latest_form}({self.frequency}:{self.state}/r{self.seen_in_row})[{self.context_key}]"

class ContextAwareTranscriptionTracker:
    """Transcription tracker that uses contextual word tracking."""
    
    def __init__(self, min_confirmed_words: int = 4, min_frequency: int = 4, 
                 min_consecutive_rounds: int = 2, context_window: int = 2):
        self.contextual_word_states: Dict[str, ContextualWordState] = {}  # context_key -> ContextualWordState
        self.confirmed_words: List[str] = []
        self.sent_words_count = 0
        self.min_confirmed_words = min_confirmed_words
        self.last_start_index = 0
        self.stall_count = 0
        self.stall_threshold = 10
        self.current_round = 0
        self.min_frequency = min_frequency
        self.min_consecutive_rounds = min_consecutive_rounds
        self.context_window = context_window  # How many words of context to use
    
    def _get_context(self, words: List[str], position: int) -> List[str]:
        """Get context for a word at given position."""
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
    
    def _create_context_key(self, word: str, context: List[str]) -> str:
        """Create a unique key for this word in context."""
        word_norm = normalize_word_for_matching(word)
        context_norm = []
        for ctx_word in context:
            if ctx_word is None:
                context_norm.append("_")
            else:
                context_norm.append(normalize_word_for_matching(ctx_word))
        return f"{word_norm}@{'-'.join(context_norm)}"
    
    def process_transcription(self, words: List[str], verbose: bool = False) -> List[str]:
        """Process transcription with context-aware word tracking."""
        if not words:
            return []
        
        self.current_round += 1
        round_id = self.current_round
        new_words_to_send = []
        
        if verbose:
            print(f"[ROUND] {round_id}: Processing {len(words)} words")
        
        # 1. Observation phase - update word states with context
        for i, word in enumerate(words):
            context = self._get_context(words, i)
            context_key = self._create_context_key(word, context)
            
            if context_key in self.contextual_word_states:
                # Update existing contextual word state
                state = self.contextual_word_states[context_key]
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
                    # Update position consistency
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
                # New contextual word
                state = ContextualWordState(word, context, frequency=1, initial_round=round_id)
                state.last_seen_pos = i
                state.pos_consistent_in_row = 1
                self.contextual_word_states[context_key] = state
            
            # State graduation phases for debug visualization
            if state.state == "unconfirmed" and state.frequency >= max(2, self.min_frequency // 2):
                state.state = "potential"
        
        # 2. Alignment phase - find where to start graduation
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
            if verbose:
                print(f"[ALIGN] Fallback boundary correction {start_index}->{min_allowed_start}")
            start_index = min_allowed_start
        
        # Track alignment stalls
        if start_index == self.last_start_index:
            self.stall_count += 1
        else:
            self.stall_count = 0
        
        # Check for stall and force progress if needed
        if self.stall_count >= self.stall_threshold:
            if verbose:
                print(f"[ALIGN] Stall detected ({self.stall_count} rounds), forcing alignment forward by 1")
            start_index = min(start_index + 1, len(self.confirmed_words))
            self.stall_count = 0
        
        self.last_start_index = start_index
        
        # 3. Graduation phase - promote contextual words to confirmed
        if verbose:
            print(f"[GRAD] Begin start_index={start_index} confirmed_len={len(self.confirmed_words)}")
        
        for i in range(start_index, len(words)):
            word = words[i]
            expected_index = len(self.confirmed_words)
            
            # Allow small position drifts due to ASR transcript instability
            position_tolerance = 1
            if abs(i - expected_index) > position_tolerance:
                if verbose:
                    reason = 'backtrack' if i < expected_index else 'forward drift'
                    print(f"[GRAD] Abort: token_index {i} != expected {expected_index} ({reason})")
                break
            elif i != expected_index:
                if verbose:
                    drift_direction = 'behind' if i < expected_index else 'ahead'
                    print(f"[GRAD] Small position drift: token index {i} vs expected {expected_index} ({drift_direction}). Allowing with tolerance.")
            
            # Get context for this word
            context = self._get_context(words, i)
            context_key = self._create_context_key(word, context)
            
            state = self.contextual_word_states.get(context_key)
            if state is None:
                if verbose:
                    print(f"[GRAD] Stop: '{word}' unseen contextual state (context_key={context_key})")
                break
            
            # Check if this contextual word should graduate
            can_graduate = state.should_graduate(
                min_frequency=self.min_frequency,
                min_consecutive=self.min_consecutive_rounds,
                require_pos_consistency=True,
                min_pos_consistent=2
            )
            
            if verbose:
                print(f"[GRAD] Eval '{word}' in context {context}: freq={state.frequency}/{self.min_frequency} consec={state.seen_in_row}/{self.min_consecutive_rounds} pos_cons={state.pos_consistent_in_row}/2 -> {'GRAD' if can_graduate else 'HOLD'}")
            
            if not can_graduate:
                break
            
            output_word = state.get_output_word()
            
            # Skip duplicates
            if (self.confirmed_words and 
                normalize_word_for_matching(self.confirmed_words[-1]) == normalize_word_for_matching(output_word)):
                if verbose:
                    print(f"[GRAD] Skip duplicate '{output_word}' (same as last confirmed)")
                continue
            
            self.confirmed_words.append(output_word)
            state.state = "confirmed"
            new_words_to_send.append(output_word)
            self.sent_words_count += 1
            
            if verbose:
                print(f"[GRAD] ✅ Graduated '{output_word}' in context {context} -> confirmed_len={len(self.confirmed_words)}")
        
        return new_words_to_send

if __name__ == "__main__":
    # Simple test
    tracker = ContextAwareTranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Test with repeated "like" in different contexts
    test_rounds = [
        ["because", "like", "obviously", "that", "spells", "that", "like", "you're"],
        ["because", "like", "obviously", "that", "spells", "that", "like", "you're"],
    ]
    
    for round_num, words in enumerate(test_rounds, 1):
        print(f"\n=== Round {round_num} ===")
        new_words = tracker.process_transcription(words, verbose=True)
        print(f"New words: {new_words}")
        print(f"Confirmed: {tracker.confirmed_words}")