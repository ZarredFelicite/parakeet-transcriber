#!/usr/bin/env python3
"""
Test the position tolerance fix for ASR transcript instability.
"""

from tmp_test_tracker import TranscriptionTracker, normalize_word_for_matching

def test_position_tolerance():
    """Test that small position drifts are handled gracefully."""
    print("=== Testing Position Tolerance ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Build up confirmed words
    initial_words = ["hello", "this", "is", "a", "test", "with", "some", "words"]
    for round_num in range(1, 3):
        tracker.current_round = round_num
        tracker.process_transcription(initial_words, verbose=False)
    
    print(f"Initial confirmed: {tracker.confirmed_words}")
    
    # Test case 1: Perfect alignment (should work)
    perfect_transcript = initial_words + ["and", "more", "content"]
    tracker.current_round = 3
    print(f"\n--- Test 1: Perfect Alignment ---")
    new_words = tracker.process_transcription(perfect_transcript, verbose=True)
    print(f"Result: {new_words}")
    
    # Test case 2: Small drift behind (should work with tolerance)
    # Simulate ASR dropping one word, causing alignment to be 1 position behind
    drift_transcript = initial_words[:-1] + ["and", "more", "content", "here"]  # Missing "words"
    tracker.current_round = 4
    print(f"\n--- Test 2: Small Drift Behind ---")
    new_words = tracker.process_transcription(drift_transcript, verbose=True)
    print(f"Result: {new_words}")
    
    # Test case 3: Large drift (should abort)
    large_drift_transcript = initial_words[:-3] + ["completely", "different", "ending"]
    tracker.current_round = 5
    print(f"\n--- Test 3: Large Drift (should abort) ---")
    new_words = tracker.process_transcription(large_drift_transcript, verbose=True)
    print(f"Result: {new_words}")
    
    print(f"\nFinal confirmed: {tracker.confirmed_words}")
    print("✅ Position tolerance test completed!")

if __name__ == "__main__":
    test_position_tolerance()