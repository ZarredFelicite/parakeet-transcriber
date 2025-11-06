#!/usr/bin/env python3
"""
Debug the 'you're' repetition issue at position 26.
"""

from tmp_test_tracker import TranscriptionTracker, normalize_word_for_matching

def debug_youre_repetition():
    """Debug why 'you're' is being repeated instead of progressing to 'buyable,'."""
    print("=== Debugging You're Repetition Issue ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Build up the confirmed words to the point just before the issue
    # Based on the logs: [To] [do] [like] [certain] [kinds] [of] [messaging] [pushes.] [Not] [only] [do] [I] [take] [issue] [with] [that] [in] [general,] [because] [like] [obviously] [that] [spells] [that] [like]
    
    confirmed_sequence = [
        "To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", 
        "Not", "only", "do", "I", "take", "issue", "with", "that", "in", 
        "general,", "because", "like", "obviously", "that", "spells", "that", "like"
    ]
    
    # Build up these confirmed words over multiple rounds
    for round_num in range(1, 4):
        tracker.current_round = round_num
        new_words = tracker.process_transcription(confirmed_sequence, verbose=False)
        if round_num == 3:
            print(f"Confirmed words after setup: {tracker.confirmed_words}")
            print(f"Length: {len(tracker.confirmed_words)}")
    
    # Now simulate the problematic transcript with "you're" -> "buyable," transition
    # This should represent chunk 32-34 from the logs
    problematic_transcript = confirmed_sequence + [
        "you're", "buyable,", "right?", "They", "can", "just", "like", "buy"
    ]
    
    print(f"\nProblematic transcript: {' '.join(problematic_transcript)}")
    print(f"Expected next words after position {len(confirmed_sequence)}: you're -> buyable, -> right?")
    
    # Process this transcript with verbose logging
    for round_num in range(4, 7):
        tracker.current_round = round_num
        print(f"\n=== Processing Round {round_num} ===")
        new_words = tracker.process_transcription(problematic_transcript, verbose=True)
        
        print(f"New words: {new_words}")
        print(f"Confirmed words: {tracker.confirmed_words}")
        print(f"Last start index: {tracker.last_start_index}")
        
        # Check the state of "you're" and "buyable,"
        youre_norm = normalize_word_for_matching("you're")
        buyable_norm = normalize_word_for_matching("buyable,")
        
        if youre_norm in tracker.word_states:
            youre_state = tracker.word_states[youre_norm]
            print(f"\n'you're' word state:")
            print(f"  Frequency: {youre_state.frequency}")
            print(f"  Consecutive rounds: {youre_state.seen_in_row}")
            print(f"  Position consistent: {youre_state.pos_consistent_in_row}")
            print(f"  Last seen position: {youre_state.last_seen_pos}")
            print(f"  State: {youre_state.state}")
        
        if buyable_norm in tracker.word_states:
            buyable_state = tracker.word_states[buyable_norm]
            print(f"\n'buyable,' word state:")
            print(f"  Frequency: {buyable_state.frequency}")
            print(f"  Consecutive rounds: {buyable_state.seen_in_row}")
            print(f"  Position consistent: {buyable_state.pos_consistent_in_row}")
            print(f"  Last seen position: {buyable_state.last_seen_pos}")
            print(f"  State: {buyable_state.state}")
        
        # Stop if we see the issue
        if len(new_words) > 0 and new_words[-1] == "you're" and len(tracker.confirmed_words) > len(confirmed_sequence) + 1:
            print("\n🚨 DETECTED: 'you're' repetition issue!")
            break

if __name__ == "__main__":
    debug_youre_repetition()