#!/usr/bin/env python3
"""
Memory Debug Tool
Test memory retrieval and search functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import get_all_memory, search_memory, get_memory
import json

def test_memory_retrieval():
    """Test memory retrieval functions"""
    print("🧠 MEMORY DEBUG TOOL")
    print("=" * 50)
    
    # Test get_all_memory
    print("\n1. Testing get_all_memory():")
    try:
        all_memory = get_all_memory()
        print(f"   Found {len(all_memory)} memory items:")
        for key, value in all_memory.items():
            print(f"   - {key}: {value}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test individual memory retrieval
    print("\n2. Testing individual memory retrieval:")
    test_keys = ['name', 'user_name', 'location', 'age', 'preferences']
    for key in test_keys:
        try:
            value = get_memory(key)
            if value:
                print(f"   - {key}: {value}")
            else:
                print(f"   - {key}: (not found)")
        except Exception as e:
            print(f"   - {key}: ERROR - {e}")
    
    # Test memory search
    print("\n3. Testing memory search:")
    search_terms = ['name', 'age', 'like']
    for term in search_terms:
        try:
            results = search_memory(term)
            print(f"   Search '{term}': {len(results)} results")
            for result in results[:3]:  # Show first 3 results
                print(f"     - {result['key']}: {result['value']} (confidence: {result['confidence']})")
        except Exception as e:
            print(f"   Search '{term}': ERROR - {e}")
    
    print("\n" + "=" * 50)

def interactive_memory_search():
    """Interactive memory search"""
    print("\n🔍 INTERACTIVE MEMORY SEARCH")
    print("Type search terms (or 'quit' to exit):")
    
    while True:
        try:
            query = input("\nSearch: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            results = search_memory(query)
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['key']}: {result['value']}")
                    print(f"   Confidence: {result['confidence']}, Updated: {result['last_updated']}")
            else:
                print("No results found.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_memory_retrieval()
    
    # Ask if user wants interactive search
    try:
        choice = input("\nWant to try interactive search? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_memory_search()
    except KeyboardInterrupt:
        pass
    
    print("\nDone! 👋")