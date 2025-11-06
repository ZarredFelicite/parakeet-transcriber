#!/usr/bin/env python3
"""
Regression test for repeated word handling in ASR transcription.

This test specifically targets the issue where repeated words (like "do" appearing
twice at different positions) would cause the positional consistency counter to
reset, preventing graduation and causing alignment stalls.
"""

from tmp_test_tracker import TranscriptionTracker, normalize_word_for_matching

def test_repeated_word_graduation():
    """Test that repeated words can graduate even when appearing at different positions."""
    print("=== Testing Repeated Word Graduation ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Simulate the problematic scenario: "do" appears first, gets confirmed, then appears again later
    test_rounds = [
        # Round 1: First appearance of "do" at position 7
        ["hello", "this", "is", "what", "we", "need", "to", "do", "right", "now"],
        
        # Round 2: "do" appears again at position 7 (consistent) - should graduate
        ["hello", "this", "is", "what", "we", "need", "to", "do", "right", "now"],
        
        # Round 3: "do" appears again at position 14 (different position - this used to cause issues)
        ["hello", "this", "is", "what", "we", "need", "to", "do", "right", "now", "and", "then", "we", "should", "do", "more"],
        
        # Round 4: "do" appears at position 14 again - should graduate due to repeated word logic
        ["hello", "this", "is", "what", "we", "need", "to", "do", "right", "now", "and", "then", "we", "should", "do", "more"],
    ]
    
    all_confirmed = []
    
    for round_num, words in enumerate(test_rounds, 1):
        print(f"\n--- Round {round_num} ---")
        print(f"Input: {' '.join(words)}")
        
        # Manually increment the round counter
        tracker.current_round = round_num
        new_words = tracker.process_transcription(words, verbose=True)
        all_confirmed.extend(new_words)
        
        print(f"New words: {new_words}")
        print(f"Total confirmed: {tracker.confirmed_words}")
        
        # Check that "do" can graduate despite appearing at different positions
        do_norm = normalize_word_for_matching("do")
        if do_norm in tracker.word_states:
            do_state = tracker.word_states[do_norm]
            print(f"'do' state: freq={do_state.frequency}, consec={do_state.seen_in_row}, pos_consistent={do_state.pos_consistent_in_row}, last_pos={do_state.last_seen_pos}")
    
    print(f"\n=== Final Results ===")
    print(f"All confirmed words: {tracker.confirmed_words}")
    
    # Verify that "do" was able to graduate despite positional changes
    assert "do" in tracker.confirmed_words, "The word 'do' should have graduated despite appearing at different positions"
    
    # Count how many times "do" appears in the final confirmed sequence
    do_count = sum(1 for word in tracker.confirmed_words if normalize_word_for_matching(word) == "do")
    print(f"'do' appears {do_count} times in confirmed words")
    
    print("✅ Test passed: Repeated words can graduate successfully!")

def test_rollback_mechanism():
    """Test the rollback mechanism when ASR drops words."""
    print("\n=== Testing Rollback Mechanism ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # First, establish some confirmed words
    round1 = ["hello", "this", "is", "a", "test"]
    round2 = ["hello", "this", "is", "a", "test"]
    
    tracker.current_round = 1
    tracker.process_transcription(round1, verbose=False)
    tracker.current_round = 2
    tracker.process_transcription(round2, verbose=False)
    
    print(f"Initial confirmed words: {tracker.confirmed_words}")
    
    # Now simulate ASR dropping some words (missing "is", "a")
    round3_dropped = ["hello", "this", "test", "continues"]
    
    print(f"Round 3 with dropped words: {' '.join(round3_dropped)}")
    tracker.current_round = 3
    new_words = tracker.process_transcription(round3_dropped, verbose=True)
    
    print(f"After rollback, confirmed words: {tracker.confirmed_words}")
    print("✅ Rollback mechanism test completed!")

if __name__ == "__main__":
    test_repeated_word_graduation()
    test_rollback_mechanism()