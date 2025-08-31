#!/usr/bin/env python3
"""
Debug the context-aware tracking with the actual streaming scenario.
"""

from tmp_test_tracker import TranscriptionTracker

def debug_context_divergence():
    """Debug the actual streaming scenario that's causing divergence."""
    print("=== Debugging Context-Aware Streaming Divergence ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Simulate the streaming chunks that lead to the divergence
    # Based on the logs, we need to build up to the point where alignment gets stuck
    
    # Build up confirmed words to chunk 31
    confirmed_sequence = [
        "To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", 
        "Not", "only", "do", "I", "take", "issue", "with", "that", "in", 
        "general,", "because", "like", "obviously", "that", "spells"
    ]
    
    # The problematic chunks 32-36
    chunks = [
        # Chunk 32: |that like you're buyable.
        confirmed_sequence + ["that", "like", "you're", "buyable."],
        # Chunk 33: |that you're buyable.  
        confirmed_sequence + ["that", "you're", "buyable."],
        # Chunk 34: |that like you're buyable, right?
        confirmed_sequence + ["that", "like", "you're", "buyable,", "right?"],
        # Chunk 35: |that you're buyable, right?
        confirmed_sequence + ["that", "you're", "buyable,", "right?"],
        # Chunk 36: |spells that you're buyable, right? (DIVERGENCE)
        confirmed_sequence + ["spells", "that", "you're", "buyable,", "right?"]
    ]
    
    # First build up the confirmed sequence
    for round_num in range(1, 4):
        tracker.current_round = round_num
        tracker.process_transcription(confirmed_sequence, verbose=False)
    
    print(f"Initial confirmed: {tracker.confirmed_words}")
    print(f"Length: {len(tracker.confirmed_words)}")
    
    # Now process the problematic chunks
    for chunk_num, chunk_words in enumerate(chunks, 32):
        print(f"\n=== Chunk {chunk_num} ===")
        print(f"Transcript: {' '.join(chunk_words[-10:])}")  # Show last 10 words
        
        tracker.current_round = chunk_num
        new_words = tracker.process_transcription(chunk_words, verbose=True)
        
        print(f"New words: {new_words}")
        print(f"Confirmed count: {len(tracker.confirmed_words)}")
        print(f"Last 5 confirmed: {tracker.confirmed_words[-5:] if len(tracker.confirmed_words) >= 5 else tracker.confirmed_words}")
        print(f"Last start index: {tracker.last_start_index}")
        
        # Check contextual word states for key words
        key_words = ["that", "spells", "like"]
        for word in key_words:
            matching_keys = [key for key in tracker.word_states.keys() if key.startswith(f"{word}@")]
            if matching_keys:
                print(f"'{word}' contexts:")
                for key in matching_keys:
                    state = tracker.word_states[key]
                    print(f"  {key}: freq={state.frequency}, consec={state.seen_in_row}, pos_cons={state.pos_consistent_in_row}")
        
        # Check if we see the divergence
        if chunk_num == 36 and len(new_words) > 0:
            if "spells" in new_words:
                print("🚨 DETECTED: 'spells' graduated instead of 'that'!")
                break

if __name__ == "__main__":
    debug_context_divergence()