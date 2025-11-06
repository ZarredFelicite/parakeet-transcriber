#!/usr/bin/env python3
"""
Debug the streaming chunk progression that leads to you're repetition.
"""

from tmp_test_tracker import TranscriptionTracker, normalize_word_for_matching

def debug_streaming_chunks():
    """Simulate the actual chunk progression from the streaming logs."""
    print("=== Debugging Streaming Chunk Progression ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Simulate the chunk progression from the logs
    # I'll focus on chunks 30-34 where the issue occurs
    
    chunks = [
        # Chunk 30: +0→25 | ... [spells] [that] [like] |you're
        {
            "chunk": 30,
            "transcript": ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I", "take", "issue", "with", "that", "in", "general,", "because", "like", "obviously", "that", "spells", "that", "like", "you're"],
            "expected_confirmed": 25,
            "expected_alignment": 25
        },
        # Chunk 31: +0→25 | ... [like] |obviously [that] [spells] [that] (you're) biased.
        {
            "chunk": 31,
            "transcript": ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I", "take", "issue", "with", "that", "in", "general,", "because", "like", "obviously", "that", "spells", "that", "you're", "biased."],
            "expected_confirmed": 25,
            "expected_alignment": 25
        },
        # Chunk 32: +1→26 | ... [spells] [that] [like] |you're buyable.
        {
            "chunk": 32,
            "transcript": ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I", "take", "issue", "with", "that", "in", "general,", "because", "like", "obviously", "that", "spells", "that", "like", "you're", "buyable."],
            "expected_confirmed": 26,
            "expected_alignment": 26
        },
        # Chunk 33: +0→26 | ... [spells] [that] [you're] |buyable.
        {
            "chunk": 33,
            "transcript": ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I", "take", "issue", "with", "that", "in", "general,", "because", "like", "obviously", "that", "spells", "that", "you're", "buyable."],
            "expected_confirmed": 26,
            "expected_alignment": 26
        },
        # Chunk 34: +1→27 | ... [spells] [that] [like] |you're (buyable,) right?
        {
            "chunk": 34,
            "transcript": ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I", "take", "issue", "with", "that", "in", "general,", "because", "like", "obviously", "that", "spells", "that", "like", "you're", "buyable,", "right?"],
            "expected_confirmed": 27,
            "expected_alignment": 27
        }
    ]
    
    # First, build up to chunk 29 (before the issue)
    initial_transcript = ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I", "take", "issue", "with", "that", "in", "general,", "because", "like", "obviously", "that", "spells", "that", "like"]
    
    for round_num in range(1, 4):
        tracker.current_round = round_num
        tracker.process_transcription(initial_transcript, verbose=False)
    
    print(f"Initial confirmed words: {tracker.confirmed_words}")
    print(f"Length: {len(tracker.confirmed_words)}")
    
    # Now process each problematic chunk
    for chunk_data in chunks:
        chunk_num = chunk_data["chunk"]
        transcript = chunk_data["transcript"]
        
        print(f"\n=== Chunk {chunk_num} ===")
        print(f"Transcript: {' '.join(transcript[-10:])}")  # Show last 10 words
        
        tracker.current_round = chunk_num
        new_words = tracker.process_transcription(transcript, verbose=True)
        
        print(f"New words: {new_words}")
        print(f"Confirmed count: {len(tracker.confirmed_words)}")
        print(f"Last 5 confirmed: {tracker.confirmed_words[-5:] if len(tracker.confirmed_words) >= 5 else tracker.confirmed_words}")
        print(f"Last start index: {tracker.last_start_index}")
        
        # Check word states for you're and buyable
        youre_norm = normalize_word_for_matching("you're")
        buyable_norm = normalize_word_for_matching("buyable,")
        
        if youre_norm in tracker.word_states:
            youre_state = tracker.word_states[youre_norm]
            print(f"'you're': freq={youre_state.frequency}, pos={youre_state.last_seen_pos}, state={youre_state.state}")
        
        if buyable_norm in tracker.word_states:
            buyable_state = tracker.word_states[buyable_norm]
            print(f"'buyable,': freq={buyable_state.frequency}, pos={buyable_state.last_seen_pos}, state={buyable_state.state}")
        
        # Check if we see the divergence pattern
        if chunk_num >= 34 and len(new_words) > 0:
            if "you're" in new_words and "buyable," not in new_words:
                print("🚨 DETECTED: you're without buyable, - this might be the issue!")

if __name__ == "__main__":
    debug_streaming_chunks()