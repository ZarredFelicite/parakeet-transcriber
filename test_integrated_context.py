#!/usr/bin/env python3
"""
Test the integrated context-aware tracking in parakeet.py.
"""

import sys
sys.path.append('/home/zarred/dev/asr2')

from parakeet import TranscriptionTracker

def test_integrated_context():
    """Test context-aware tracking integrated into main parakeet.py."""
    print("=== Testing Integrated Context-Aware Tracking ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Test the specific divergence scenario
    test_sequence = [
        "because", "like", "obviously", "that", "spells", "that", "like", "you're", "buyable"
    ]
    
    print(f"Testing sequence: {' '.join(test_sequence)}")
    print("Focus: Two different 'like' contexts should be tracked separately")
    
    # Process over multiple rounds
    for round_num in range(1, 4):
        print(f"\n=== Round {round_num} ===")
        new_words = tracker.process_transcription(test_sequence, verbose=True)
        print(f"New words: {new_words}")
        print(f"Confirmed: {tracker.confirmed_words}")
        
        if round_num == 3:
            print(f"\n=== Context Analysis ===")
            print(f"Total contextual word states: {len(tracker.word_states)}")
            
            # Show all contextual keys
            like_contexts = []
            for key in tracker.word_states.keys():
                if key.startswith("like@"):
                    like_contexts.append(key)
                    state = tracker.word_states[key]
                    print(f"'like' context: {key}")
                    print(f"  Frequency: {state.frequency}, State: {state.state}")
            
            print(f"\nFound {len(like_contexts)} different 'like' contexts")
            
            # Check if both "like" instances are properly handled
            expected_contexts = [
                "like@because-like-obviously",  # First "like"
                "like@that-like-youre"          # Second "like"
            ]
            
            for expected in expected_contexts:
                if any(expected in key for key in like_contexts):
                    print(f"✅ Found expected context pattern: {expected}")
                else:
                    print(f"❌ Missing expected context pattern: {expected}")
    
    print(f"\n=== Final Results ===")
    print(f"Final confirmed: {tracker.confirmed_words}")
    
    # Check if the sequence is correct
    expected_sequence = ["because", "like", "obviously", "that", "spells", "that", "like", "you're", "buyable"]
    if tracker.confirmed_words == expected_sequence:
        print("✅ SUCCESS: Context-aware tracking produced correct sequence!")
    else:
        print("❌ Sequence mismatch - context tracking needs adjustment")
        print(f"Expected: {expected_sequence}")
        print(f"Got:      {tracker.confirmed_words}")

if __name__ == "__main__":
    test_integrated_context()