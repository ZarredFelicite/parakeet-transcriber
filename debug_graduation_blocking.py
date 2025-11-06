#!/usr/bin/env python3
"""
Debug what's blocking graduation when words have sufficient frequency.
"""

from tmp_test_tracker import TranscriptionTracker

def debug_graduation_blocking():
    """Debug graduation blocking with high frequency words."""
    print("=== Debugging Graduation Blocking ===")
    
    tracker = TranscriptionTracker(min_frequency=4, min_consecutive_rounds=2)
    
    # Simulate many chunks to build up frequency=4
    base_words = ["To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", "Not", "only", "do", "I", "take", "issue", "with", "that", "in", "general,", "because", "like", "obviously", "that", "spells"]
    
    # Process many rounds to build up frequency
    for round_num in range(1, 8):
        tracker.current_round = round_num
        new_words = tracker.process_transcription(base_words, verbose=False)
        
        if round_num % 2 == 0:  # Check every 2 rounds
            print(f"\n=== After Round {round_num} ===")
            print(f"Confirmed words: {len(tracker.confirmed_words)}")
            
            # Check specific words that should have high frequency
            test_words = ["to", "do", "like"]
            for word in test_words:
                if word in tracker.word_states:
                    state = tracker.word_states[word]
                    can_graduate = state.should_graduate(
                        min_frequency=tracker.min_frequency,
                        min_consecutive=tracker.min_consecutive_rounds,
                        require_pos_consistency=True,
                        min_pos_consistent=2,
                        confirmed_words=tracker.confirmed_words
                    )
                    print(f"  '{word}': freq={state.frequency}/{tracker.min_frequency}, consec={state.seen_in_row}/{tracker.min_consecutive_rounds}, pos_cons={state.pos_consistent_in_row}/2, can_grad={can_graduate}")
                    
                    # Detailed graduation check
                    freq_ok = state.frequency >= tracker.min_frequency
                    consec_ok = state.seen_in_row >= tracker.min_consecutive_rounds
                    pos_ok = state.pos_consistent_in_row >= 2
                    
                    print(f"    freq_ok={freq_ok}, consec_ok={consec_ok}, pos_ok={pos_ok}")
                    
                    if not can_graduate:
                        if not freq_ok:
                            print(f"    ❌ BLOCKED: Insufficient frequency ({state.frequency} < {tracker.min_frequency})")
                        elif not consec_ok:
                            print(f"    ❌ BLOCKED: Insufficient consecutive rounds ({state.seen_in_row} < {tracker.min_consecutive_rounds})")
                        elif not pos_ok:
                            print(f"    ❌ BLOCKED: Insufficient position consistency ({state.pos_consistent_in_row} < 2)")
        
        if len(tracker.confirmed_words) > 0:
            print(f"\n✅ Graduation started at round {round_num}!")
            break
    
    if len(tracker.confirmed_words) == 0:
        print(f"\n❌ No graduation after {round_num} rounds")
        print("This suggests a fundamental issue with graduation logic")

if __name__ == "__main__":
    debug_graduation_blocking()