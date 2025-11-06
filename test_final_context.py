#!/usr/bin/env python3
"""
Final test of context-aware tracking with the updated test tracker.
"""

from tmp_test_tracker import TranscriptionTracker

def test_final_context():
    """Test the final context-aware implementation."""
    print("=== Final Context-Aware Test ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Test the exact divergence scenario
    test_sequence = [
        "because", "like", "obviously", "that", "spells", "that", "like", "you're", "buyable"
    ]
    
    print(f"Testing sequence: {' '.join(test_sequence)}")
    print("Expected: Both 'like' instances should graduate in their respective contexts")
    
    # Process over multiple rounds
    for round_num in range(1, 4):
        print(f"\n=== Round {round_num} ===")
        new_words = tracker.process_transcription(test_sequence, verbose=True)
        print(f"New words: {new_words}")
        print(f"Confirmed: {tracker.confirmed_words}")
        
        if round_num == 3:
            print(f"\n=== Context Analysis ===")
            print(f"Total contextual word states: {len(tracker.word_states)}")
            
            # Show contextual keys for 'like'
            like_keys = [key for key in tracker.word_states.keys() if key.startswith("like@")]
            print(f"'like' contextual keys: {like_keys}")
            
            for key in like_keys:
                state = tracker.word_states[key]
                print(f"  {key}: freq={state.frequency}, state={state.state}")
    
    print(f"\n=== Final Results ===")
    print(f"Final confirmed: {tracker.confirmed_words}")
    
    # Check if the sequence is correct
    if len(tracker.confirmed_words) >= 7 and tracker.confirmed_words[6] == "like":
        print("✅ SUCCESS: Second 'like' graduated correctly!")
    else:
        print("❌ Second 'like' did not graduate as expected")
    
    # Check for both 'like' instances
    like_count = tracker.confirmed_words.count("like")
    print(f"Total 'like' instances in confirmed words: {like_count}")
    
    if like_count == 2:
        print("✅ SUCCESS: Both 'like' instances graduated!")
    else:
        print(f"❌ Expected 2 'like' instances, got {like_count}")

if __name__ == "__main__":
    test_final_context()