#!/usr/bin/env python3
"""
Debug why no words are graduating to confirmed status.
"""

from tmp_test_tracker import TranscriptionTracker

def debug_no_graduation():
    """Debug the graduation issue in streaming scenario."""
    print("=== Debugging No Graduation Issue ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Simulate the first few chunks to see what's happening
    chunks = [
        # Chunk 11
        ["to", "to", "do", "like", "certain", "kind", "of"],
        # Chunk 12  
        ["To", "to", "do", "like", "certain", "kind", "of", "messages"],
        # Chunk 13
        ["To", "to", "do", "like", "certain", "kind", "of", "messaging", "push."],
        # Chunk 14
        ["to", "to", "do", "like", "certain", "kind", "of", "messaging", "pushes."],
        # Chunk 15
        ["To", "do", "like", "certain", "kind", "of", "messaging", "pushes."],
        # Chunk 16
        ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I"],
    ]
    
    for chunk_num, words in enumerate(chunks, 11):
        print(f"\n=== Chunk {chunk_num} ===")
        print(f"Input: {' '.join(words)}")
        
        tracker.current_round = chunk_num
        new_words = tracker.process_transcription(words, verbose=True)
        
        print(f"New words: {new_words}")
        print(f"Confirmed: {tracker.confirmed_words}")
        print(f"Total word states: {len(tracker.word_states)}")
        
        # Show some example word states
        print(f"\nSample word states:")
        count = 0
        for key, state in tracker.word_states.items():
            if count < 5:  # Show first 5
                print(f"  {key}: freq={state.frequency}, consec={state.seen_in_row}, pos_cons={state.pos_consistent_in_row}")
                count += 1
        
        # Check specific words
        target_words = ["to", "do", "like"]
        for word in target_words:
            matching_keys = [key for key in tracker.word_states.keys() if key.startswith(f"{word}@")]
            if matching_keys:
                print(f"\n'{word}' contexts ({len(matching_keys)} total):")
                for key in matching_keys[:3]:  # Show first 3
                    state = tracker.word_states[key]
                    can_graduate = state.should_graduate(
                        min_frequency=max(1, tracker.min_frequency - 1),
                        min_consecutive=max(1, tracker.min_consecutive_rounds - 1),
                        require_pos_consistency=True,
                        min_pos_consistent=2,
                        confirmed_words=tracker.confirmed_words
                    )
                    print(f"  {key}: freq={state.frequency}, consec={state.seen_in_row}, can_grad={can_graduate}")
        
        if len(tracker.confirmed_words) > 0:
            print(f"✅ Words started graduating!")
            break
        elif chunk_num >= 15:
            print(f"❌ No words graduated after {chunk_num - 10} chunks")
            break

if __name__ == "__main__":
    debug_no_graduation()