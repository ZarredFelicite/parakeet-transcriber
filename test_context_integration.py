#!/usr/bin/env python3
"""
Test context-aware tracking by updating the test tracker.
"""

from tmp_test_tracker import TranscriptionTracker, normalize_word_for_matching

# Add context-aware methods to the existing test tracker
def add_context_methods(tracker):
    """Add context-aware methods to existing tracker."""
    tracker.context_window = 2
    tracker.use_context_aware = True
    
    def _get_context(self, words, position):
        """Get context for a word at given position."""
        if not self.use_context_aware:
            return None
            
        context = []
        
        # Add previous words (up to context_window)
        for i in range(max(0, position - self.context_window), position):
            context.append(words[i] if i < len(words) else None)
        
        # Add the word itself
        context.append(words[position] if position < len(words) else None)
        
        # Add next words (up to 1 for now)
        for i in range(position + 1, min(len(words), position + 2)):
            context.append(words[i] if i < len(words) else None)
        
        # Pad with None if needed to maintain consistent context size
        while len(context) < self.context_window + 2:  # prev + word + next
            context.append(None)
        
        return context
    
    # Bind the method to the tracker instance
    import types
    tracker._get_context = types.MethodType(_get_context, tracker)

def test_context_integration():
    """Test context-aware tracking with the test tracker."""
    print("=== Testing Context-Aware Integration ===")
    
    tracker = TranscriptionTracker(min_frequency=2, min_consecutive_rounds=2)
    add_context_methods(tracker)
    
    # Test the divergence scenario
    test_sequence = [
        "because", "like", "obviously", "that", "spells", "that", "like", "you're"
    ]
    
    print(f"Testing sequence: {' '.join(test_sequence)}")
    
    # Manually test context extraction
    print(f"\n=== Context Extraction Test ===")
    for i, word in enumerate(test_sequence):
        context = tracker._get_context(test_sequence, i)
        context_key = f"{normalize_word_for_matching(word)}@{'-'.join([normalize_word_for_matching(w) if w else '_' for w in context])}"
        print(f"Position {i}: '{word}' -> context: {context} -> key: {context_key}")
    
    # Check that the two "like" instances have different keys
    like_positions = [i for i, word in enumerate(test_sequence) if word == "like"]
    print(f"\n=== 'like' Context Analysis ===")
    
    for pos in like_positions:
        context = tracker._get_context(test_sequence, pos)
        context_key = f"like@{'-'.join([normalize_word_for_matching(w) if w else '_' for w in context])}"
        print(f"'like' at position {pos}: context={context} key={context_key}")
    
    if len(set([f"like@{'-'.join([normalize_word_for_matching(w) if w else '_' for w in tracker._get_context(test_sequence, pos)])}" for pos in like_positions])) == len(like_positions):
        print("✅ SUCCESS: Different 'like' instances have unique context keys!")
    else:
        print("❌ FAILURE: 'like' instances have same context keys")

if __name__ == "__main__":
    test_context_integration()