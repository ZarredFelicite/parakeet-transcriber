#!/usr/bin/env python3
"""
Test context-aware tracking with the specific divergence scenario.
"""

from context_aware_tracker import ContextAwareTranscriptionTracker

def test_context_divergence():
    """Test the specific divergence: 'spells' vs 'like' at position 22."""
    print("=== Testing Context-Aware Divergence Fix ===")
    
    tracker = ContextAwareTranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    
    # Build up the confirmed sequence to the point of divergence
    # From logs: [To] [do] [like] [certain] [kinds] [of] [messaging] [pushes.] [Not] [only] [do] [I] [take] [issue] [with] [that] [in] [general,] [because] [obviously] [that]
    
    confirmed_sequence = [
        "To", "do", "like", "certain", "kinds", "of", "messaging", "pushes.", 
        "Not", "only", "do", "I", "take", "issue", "with", "that", "in", 
        "general,", "because", "obviously", "that"
    ]
    
    # The problematic part: "spells that like" 
    # Expected: [because] [obviously] [that] [spells] [that] [like] [you're]
    # Streaming got: [because] [obviously] [that] [like] [MISSING] [MISSING] [MISSING]
    
    problematic_sequence = confirmed_sequence + ["spells", "that", "like", "you're", "buyable"]
    
    print(f"Testing sequence: {' '.join(problematic_sequence)}")
    print(f"Focus on: ...obviously that spells that like you're...")
    
    # Process over multiple rounds to build up context
    for round_num in range(1, 4):
        print(f"\n=== Round {round_num} ===")
        new_words = tracker.process_transcription(problematic_sequence, verbose=True)
        print(f"New words: {new_words}")
        print(f"Confirmed: {tracker.confirmed_words}")
        
        if round_num == 3:
            # Check the specific contextual states
            print(f"\n=== Contextual Word Analysis ===")
            
            # Check "like" in different contexts
            like_contexts = []
            for key, state in tracker.contextual_word_states.items():
                if "like@" in key:
                    like_contexts.append((key, state))
                    print(f"'like' context: {key}")
                    print(f"  Frequency: {state.frequency}, State: {state.state}")
            
            # Check "spells" context
            spells_contexts = []
            for key, state in tracker.contextual_word_states.items():
                if "spells@" in key:
                    spells_contexts.append((key, state))
                    print(f"'spells' context: {key}")
                    print(f"  Frequency: {state.frequency}, State: {state.state}")
            
            print(f"\nFound {len(like_contexts)} different 'like' contexts")
            print(f"Found {len(spells_contexts)} different 'spells' contexts")
    
    print(f"\n=== Final Results ===")
    print(f"Final confirmed: {tracker.confirmed_words}")
    print(f"Expected at position 22: 'spells'")
    if len(tracker.confirmed_words) > 22:
        print(f"Actual at position 22: '{tracker.confirmed_words[22]}'")
        if tracker.confirmed_words[22] == "spells":
            print("✅ SUCCESS: Context-aware tracking resolved the divergence!")
        else:
            print("❌ Still diverged, but context tracking is working")
    else:
        print(f"Only {len(tracker.confirmed_words)} words confirmed so far")

if __name__ == "__main__":
    test_context_divergence()