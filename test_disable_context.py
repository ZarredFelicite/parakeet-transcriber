#!/usr/bin/env python3
"""
Test with context-aware tracking disabled to see if that fixes graduation.
"""

from tmp_test_tracker import TranscriptionTracker

def test_disable_context():
    """Test with context-aware tracking disabled."""
    print("=== Testing with Context-Aware Disabled ===")
    
    # Test both with and without context-aware tracking
    configs = [
        {"name": "Context-Aware ON", "use_context": True},
        {"name": "Context-Aware OFF", "use_context": False}
    ]
    
    for config in configs:
        print(f"\n=== {config['name']} ===")
        
        tracker = TranscriptionTracker(min_frequency=4, min_consecutive_rounds=2)
        tracker.use_context_aware = config['use_context']
        
        # Simulate streaming chunks
        chunks = [
            ["to", "to", "do", "like", "certain", "kind", "of"],
            ["To", "to", "do", "like", "certain", "kind", "of", "messages"],
            ["To", "to", "do", "like", "certain", "kind", "of", "messaging", "push."],
            ["to", "to", "do", "like", "certain", "kind", "of", "messaging", "pushes."],
            ["To", "do", "like", "certain", "kind", "of", "messaging", "pushes."],
        ]
        
        for chunk_num, words in enumerate(chunks, 11):
            tracker.current_round = chunk_num
            new_words = tracker.process_transcription(words, verbose=False)
            
            if len(tracker.confirmed_words) > 0:
                print(f"✅ First graduation at chunk {chunk_num}: {new_words}")
                print(f"Confirmed: {tracker.confirmed_words}")
                break
        else:
            print(f"❌ No graduation after {len(chunks)} chunks")
            
            # Show word states
            print("Sample word states:")
            for key, state in list(tracker.word_states.items())[:5]:
                print(f"  {key}: freq={state.frequency}, consec={state.seen_in_row}")

if __name__ == "__main__":
    test_disable_context()