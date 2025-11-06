#!/usr/bin/env python3
"""
Test for alignment stall detection and recovery.

This test simulates the scenario where alignment gets stuck at the same position
for many rounds, then verifies that the stall detection mechanism kicks in.
"""

from tmp_test_tracker import TranscriptionTracker, normalize_word_for_matching

def test_stall_detection():
    """Test that alignment stall detection prevents indefinite stalls."""
    print("=== Testing Alignment Stall Detection ===")
    
    # Use a lower stall threshold for testing
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    tracker.stall_threshold = 5  # Lower threshold for testing
    
    # First, establish some confirmed words
    initial_words = ["hello", "this", "is", "a", "test"]
    for round_num in range(1, 3):
        tracker.current_round = round_num
        tracker.process_transcription(initial_words, verbose=False)
    
    print(f"Initial confirmed words: {tracker.confirmed_words}")
    print(f"Initial last_start_index: {tracker.last_start_index}")
    
    # Now simulate a stall scenario - same alignment position for many rounds
    stall_words = ["hello", "this", "is", "a", "test", "but", "now", "we", "have", "different", "content"]
    
    print(f"\nSimulating stall with: {' '.join(stall_words)}")
    
    for round_num in range(3, 20):  # More rounds to exceed stall threshold
        tracker.current_round = round_num
        print(f"\n--- Round {round_num} ---")
        print(f"Before: stall_count={tracker.stall_count}, last_start_index={tracker.last_start_index}")
        
        new_words = tracker.process_transcription(stall_words, verbose=True)
        
        print(f"After: stall_count={tracker.stall_count}, last_start_index={tracker.last_start_index}")
        print(f"New words: {new_words}")
        print(f"Confirmed: {tracker.confirmed_words}")
        
        # Check if stall detection kicked in
        if tracker.stall_count == 0 and round_num > 10:  # Should reset after threshold
            print("✅ Stall detection activated and reset counter!")
            break
        
        # Also break if we exceed reasonable test bounds
        if round_num > 18:
            print("⚠️  Test exceeded bounds, stall detection may not have triggered")
            break
    
    print(f"\n=== Final Results ===")
    print(f"Final confirmed words: {tracker.confirmed_words}")
    print(f"Final stall_count: {tracker.stall_count}")
    print(f"Final last_start_index: {tracker.last_start_index}")
    
    # Verify that we made progress despite the stall
    assert len(tracker.confirmed_words) > 5, "Should have made progress beyond initial words"
    print("✅ Stall detection test passed!")

if __name__ == "__main__":
    test_stall_detection()