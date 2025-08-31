#!/usr/bin/env python3
"""
Debug the exact alignment issue with the "do" word at position 10.
"""

from tmp_test_tracker import TranscriptionTracker, normalize_word_for_matching

def debug_alignment_issue():
    """Debug why alignment gets stuck at position 10 with 'do'."""
    print("=== Debugging Alignment Issue ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Simulate the exact scenario from the logs
    # First, build up the confirmed words: [To] [do] [like] [certain] [kinds] [of] [messaging] [pushes.] [Not] [only]
    
    # Round 1-2: Build up initial words
    initial_transcript = ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only"]
    
    for round_num in range(1, 3):
        tracker.current_round = round_num
        new_words = tracker.process_transcription(initial_transcript, verbose=False)
        print(f"Round {round_num}: {new_words}")
    
    print(f"Confirmed words after setup: {tracker.confirmed_words}")
    print(f"Length: {len(tracker.confirmed_words)}")
    
    # Now simulate the problematic transcript with "do" at position 10
    problematic_transcript = [
        "To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only",  # positions 0-9
        "do", "I", "take", "issue", "with", "that", "in", "general,", "because", "like"       # positions 10-19
    ]
    
    print(f"\nProblematic transcript: {' '.join(problematic_transcript)}")
    print(f"Word at position 10: '{problematic_transcript[10]}'")
    
    # Process this transcript with verbose logging
    tracker.current_round = 3
    print(f"\n=== Processing Round 3 ===")
    new_words = tracker.process_transcription(problematic_transcript, verbose=True)
    
    print(f"\nResults:")
    print(f"New words: {new_words}")
    print(f"Confirmed words: {tracker.confirmed_words}")
    print(f"Last start index: {tracker.last_start_index}")
    
    # Check the state of the word "do"
    do_norm = normalize_word_for_matching("do")
    if do_norm in tracker.word_states:
        do_state = tracker.word_states[do_norm]
        print(f"\n'do' word state:")
        print(f"  Frequency: {do_state.frequency}")
        print(f"  Consecutive rounds: {do_state.seen_in_row}")
        print(f"  Position consistent: {do_state.pos_consistent_in_row}")
        print(f"  Last seen position: {do_state.last_seen_pos}")
        print(f"  State: {do_state.state}")
        
        # Check if it should graduate
        should_grad = do_state.should_graduate(
            min_frequency=tracker.min_frequency,
            min_consecutive=tracker.min_consecutive_rounds,
            require_pos_consistency=True,
            min_pos_consistent=2,
            confirmed_words=tracker.confirmed_words
        )
        print(f"  Should graduate: {should_grad}")

if __name__ == "__main__":
    debug_alignment_issue()