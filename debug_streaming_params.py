#!/usr/bin/env python3
"""
Debug with streaming parameters that match the actual system.
"""

from tmp_test_tracker import TranscriptionTracker

def debug_streaming_params():
    """Debug with parameters that match the actual streaming system."""
    print("=== Debugging with Streaming Parameters ===")
    
    # Try to match the actual streaming system parameters
    # The debug shows 'to(4)' suggesting min_frequency=4
    tracker = TranscriptionTracker(min_frequency=4, min_consecutive_rounds=2)
    
    print(f"Using parameters: min_frequency={tracker.min_frequency}, min_consecutive={tracker.min_consecutive_rounds}")
    print(f"Context-aware: {tracker.use_context_aware}")
    
    # Simulate more chunks to reach frequency=4
    chunks = [
        ["to", "to", "do", "like", "certain", "kind", "of"],
        ["To", "to", "do", "like", "certain", "kind", "of", "messages"],
        ["To", "to", "do", "like", "certain", "kind", "of", "messaging", "push."],
        ["to", "to", "do", "like", "certain", "kind", "of", "messaging", "pushes."],
        ["To", "do", "like", "certain", "kind", "of", "messaging", "pushes."],
        ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I"],
        ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I", "take"],
        ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I", "take", "issue"],
    ]
    
    for chunk_num, words in enumerate(chunks, 11):
        print(f"\n=== Chunk {chunk_num} ===")
        print(f"Input: {' '.join(words[:10])}{'...' if len(words) > 10 else ''}")
        
        tracker.current_round = chunk_num
        new_words = tracker.process_transcription(words, verbose=False)  # Less verbose for clarity
        
        print(f"New words: {new_words}")
        print(f"Confirmed: {len(tracker.confirmed_words)} words")
        
        # Check key word states
        key_words = ["to", "do", "like"]
        for word in key_words:
            matching_keys = [key for key in tracker.word_states.keys() if key.startswith(f"{word}@")]
            if matching_keys:
                # Find the highest frequency context for this word
                best_state = max([tracker.word_states[key] for key in matching_keys], key=lambda s: s.frequency)
                can_graduate = best_state.should_graduate(
                    min_frequency=max(1, tracker.min_frequency - 1),
                    min_consecutive=max(1, tracker.min_consecutive_rounds - 1),
                    require_pos_consistency=True,
                    min_pos_consistent=2,
                    confirmed_words=tracker.confirmed_words
                )
                print(f"  '{word}' best: freq={best_state.frequency}/{tracker.min_frequency}, consec={best_state.seen_in_row}/{tracker.min_consecutive_rounds}, can_grad={can_graduate}")
        
        if len(tracker.confirmed_words) > 0:
            print(f"✅ First graduation at chunk {chunk_num}!")
            break
    
    # Check if context-aware tracking is causing too much fragmentation
    print(f"\n=== Context Fragmentation Analysis ===")
    word_context_counts = {}
    for key in tracker.word_states.keys():
        if "@" in key:
            word = key.split("@")[0]
            word_context_counts[word] = word_context_counts.get(word, 0) + 1
    
    print("Words with multiple contexts:")
    for word, count in sorted(word_context_counts.items()):
        if count > 1:
            print(f"  '{word}': {count} different contexts")
    
    # Check if we should disable context-aware tracking
    if len(tracker.confirmed_words) == 0:
        print(f"\n=== Testing without context-aware tracking ===")
        tracker_no_context = TranscriptionTracker(min_frequency=4, min_consecutive_rounds=2)
        tracker_no_context.use_context_aware = False
        
        # Test the same chunks
        for chunk_num, words in enumerate(chunks[:4], 11):
            tracker_no_context.current_round = chunk_num
            new_words = tracker_no_context.process_transcription(words, verbose=False)
            if len(tracker_no_context.confirmed_words) > 0:
                print(f"✅ Without context-aware: graduation at chunk {chunk_num}!")
                print(f"Confirmed: {tracker_no_context.confirmed_words}")
                break

if __name__ == "__main__":
    debug_streaming_params()