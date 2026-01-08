#!/usr/bin/env python3
"""
Test search_conversations function specifically
"""
import sys
sys.path.append('.')

from memory import search_conversations
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

print("Testing search_conversations function...")

# Test with different inputs
test_queries = [
    "hello",
    "how are you",
    "",
    None,
    123
]

for query in test_queries:
    print(f"\nTesting query: {repr(query)}")
    try:
        result = search_conversations(query)
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        print(f"Length: {len(result) if result else 0}")
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")