import re
import json
from memory import save_memory, get_memory

def extract_comprehensive_info(text, current_memory):
    """Extract all possible information from natural conversation (excluding names)"""
    text_lower = text.strip().lower()
    info = {}
    
    # Universal patterns for any information type (excluding name patterns)
    universal_patterns = [
        # Direct statements (excluding name-related ones)
        r"i am\s+(\d+)\s*(?:years?\s*old)?",  # Age only
        r"i'm\s+(\d+)\s*(?:years?\s*old)?",   # Age only
        r"my\s+(age|location|job|work|occupation)\s+is\s+([^.!?\n]+)",
        r"i have\s+([^.!?\n]+)",
        r"i live\s+(?:in\s+)?([^.!?\n]+)",
        r"i work\s+(?:as\s+|at\s+)?([^.!?\n]+)",
        r"i study\s+([^.!?\n]+)",
        r"i like\s+([^.!?\n]+)",
        r"i love\s+([^.!?\n]+)",
        r"i hate\s+([^.!?\n]+)",
        r"i prefer\s+([^.!?\n]+)",
        r"i enjoy\s+([^.!?\n]+)",
        r"i can\s+([^.!?\n]+)",
        r"i know\s+([^.!?\n]+)",
        r"i speak\s+([^.!?\n]+)",
        
        # Corrections and clarifications
        r"actually\s+i\s+(?:am|live|work|study|like|love)\s+([^.!?\n]+)",
        r"no[,\s]+i\s+(?:am|live|work|study|like|love)\s+([^.!?\n]+)",
        
        # Possessive statements (excluding name)
        r"my\s+(hobby|hobbies|favorite|favourite|pet|car|house|family)\s+([^.!?\n]+)",
        
        # Time and dates
        r"(\w+day)\s+([^.!?\n]+)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+([^.!?\n]+)",
        r"(\d{1,2})[st|nd|rd|th]*\s+of\s+(\w+)",
        
        # Relationships (excluding name extraction)
        r"my\s+(friend|brother|sister|mother|father|parent|family|wife|husband|partner)\s+(?:is|lives|works|likes)\s+([^.!?\n]+)",
    ]
    
    # Extract using universal patterns
    for pattern in universal_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            groups = match.groups()
            if len(groups) == 1:
                value = groups[0].strip()
                if value and len(value) > 1:
                    # Determine key based on context
                    key = determine_info_type(match.group(0), value)
                    if key:
                        info[key] = value
            elif len(groups) == 2:
                key, value = groups[0].strip(), groups[1].strip()
                if key and value and len(value) > 1:
                    info[key] = value
    
    # Skip name extraction - names come from face recognition
    
    return info

def extract_name_specifically(text):
    """Dedicated name extraction with comprehensive patterns"""
    name_patterns = [
        r"hi[,\s]+i'?m\s+([a-zA-Z]+)",
        r"hello[,\s]+i'?m\s+([a-zA-Z]+)",
        r"hey[,\s]+i'?m\s+([a-zA-Z]+)",
        r"i'?m\s+([a-zA-Z]+)(?:\s|$|[.!?])",
        r"my name is\s+([a-zA-Z]+)",
        r"call me\s+([a-zA-Z]+)",
        r"i am\s+([a-zA-Z]+)(?:\s|$|[.!?])",
        r"this is\s+([a-zA-Z]+)",
        r"([a-zA-Z]+)\s+here",
        r"it'?s\s+([a-zA-Z]+)",
        r"name'?s\s+([a-zA-Z]+)",
        r"actually\s+i'?m\s+([a-zA-Z]+)",
        r"no[,\s]+i'?m\s+([a-zA-Z]+)",
        r"i'?m\s+actually\s+([a-zA-Z]+)",
        r"who is\s+\w+\?*\s*i'?m\s+([a-zA-Z]+)",
        r"not\s+\w+[,\s]+i'?m\s+([a-zA-Z]+)",
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).capitalize()
            # Filter out common non-names
            if name.lower() not in ['not', 'am', 'is', 'are', 'was', 'were', 'here', 'there', 'good', 'fine', 'okay', 'ok', 'yes', 'no']:
                return name
    return None

def determine_info_type(full_match, value):
    """Determine what type of information this is based on context"""
    full_match = full_match.lower()
    value = value.lower().strip()
    
    # Skip if value is too generic or question words
    if value in ['good', 'fine', 'okay', 'ok', 'yes', 'no', 'here', 'there', 'date', 'time', 'today', 'tomorrow', 'yesterday']:
        return None
    
    # Skip if it's clearly a question pattern
    if any(word in full_match for word in ['what is', 'what', 'when', 'where', 'how', 'why']):
        return None
    
    # Age detection
    if re.match(r'^\d{1,3}(\s+years?\s+old)?$', value):
        return 'age'
    
    # Location detection
    if any(word in full_match for word in ['from', 'live', 'in']):
        return 'location'
    
    # Job/occupation detection
    if any(word in full_match for word in ['work', 'job', 'profession']):
        return 'occupation'
    
    # Study/education detection
    if 'study' in full_match:
        return 'education'
    
    # Language detection
    if 'speak' in full_match:
        return 'languages'
    
    # Preferences
    if any(word in full_match for word in ['like', 'love', 'enjoy', 'prefer']):
        return 'preferences'
    
    # Skills
    if any(word in full_match for word in ['can', 'know how', 'able']):
        return 'skills'
    
    # Default to generic info
    return 'info'

def intelligent_memory_save(user_input, intent, current_memory):
    """Comprehensive memory extraction from any natural conversation (excluding names)"""
    memory_ops = []
    
    # Extract all possible information (excluding names)
    extracted_info = extract_comprehensive_info(user_input, current_memory)
    
    # Process each piece of extracted information
    for key, value in extracted_info.items():
        # Skip name-related keys since names come from face recognition
        if key in ['name', 'user_name', 'my_name', 'called']:
            continue
            
        current_value = current_memory.get(key)
        
        # Determine confidence based on key type and context
        confidence = 0.7
        if any(phrase in user_input.lower() for phrase in ['actually', 'no', 'not', 'correction']):
            confidence = 0.9  # Higher confidence for corrections
        
        # Only save if new or different
        if not current_value or str(current_value).lower() != str(value).lower():
            memory_ops.append({
                "key": key,
                "value": value,
                "confidence": confidence
            })
    
    return memory_ops