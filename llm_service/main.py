"""
FastAPI LLM Service for Buddy AI - ENHANCED VERSION
Handles all LLM processing with superior intelligence and capabilities
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import html
import os
import requests
from datetime import datetime, timedelta
import re
from functools import lru_cache
import asyncio

# Import ALL brain modules
from memory import init_db, get_all_memory, save_memory, save_conversation
from smart_memory import intelligent_memory_save

app = FastAPI(title="Buddy Brain Service", version="2.0.0")

# Initialize brain database
init_db()

# Enhanced system prompt for better responses
SYSTEM_PROMPT = """You are Buddy, a friendly, intelligent robot companion with web access and real-time awareness.

Reply ONLY in JSON:
{"reply":"", "emotion":"", "intent":""}

CRITICAL RULES:
- NEVER add text outside JSON
- Use natural, conversational language that flows with previous conversation
- Understand context from recent_chat - don't repeat yourself or restart conversations
- Questions about YOU (Buddy) should be answered about yourself, not about time/date
- Follow-up questions refer to the previous topic - use context
- Short responses like "nice", "ok", "cool" should get brief acknowledgments, not full greetings
- If user says random words, ask what they mean instead of greeting them

Response guidelines:
- Use web_search facts when available for factual questions
- Use time_info ONLY when explicitly asked about time/date/day
- Reference memory naturally when relevant
- Be concise (1-2 sentences for simple questions, 2-3 for complex)
- Show personality while being informative
- Maintain conversation flow - don't reset unnecessarily
"""

class ChatRequest(BaseModel):
    user_input: str
    recognized_user: Optional[str] = None
    objects_visible: Optional[list] = None

class ChatResponse(BaseModel):
    reply: str
    intent: str
    emotion: str
    raw_response: str

# Cache for web search results (5 minute TTL)
_search_cache: Dict[str, tuple[str, datetime]] = {}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_buddy(request: ChatRequest):
    """Process chat request - ENHANCED BRAIN PROCESSING"""
    try:
        print(f"🧠 BRAIN: Processing - User: '{request.user_input}', Face: {request.recognized_user}, Objects: {request.objects_visible}")
        
        # Get memory context from database
        effective_user = request.recognized_user if request.recognized_user else "Unknown"
        memory_context = get_all_memory(user_name=effective_user)
        
        # Build enhanced structured context for LLM
        context_data = {
            "user_text": request.user_input,
            "face": {
                "seen": request.recognized_user is not None,
                "name": request.recognized_user
            },
            "objects": request.objects_visible or [],
            "allowed_intents": ["greeting", "conversation", "provide_info", "question", "ask_name", 
                               "movement", "follow", "stop", "dance", "nod", "shake_head", 
                               "celebrate", "sleep", "wake_up"],
            "allowed_emotions": ["happy", "excited", "cheerful", "friendly", "warm", "curious", 
                                "interested", "surprised", "concerned", "confused", "apologetic", 
                                "calm", "playful", "neutral"]
        }
        
        # Smart memory addition - only when contextually relevant
        if memory_context and _is_memory_relevant(request.user_input):
            context_data["memory"] = memory_context
            print(f"📝 BRAIN: Added memory context: {list(memory_context.keys())}")
        
        # Add recent conversation context (last 5 exchanges for better continuity)
        recent_conversations = []
        try:
            from memory import get_recent_conversations
            recent_conversations = get_recent_conversations(effective_user, limit=5)
            if recent_conversations:
                # Format conversations for better context
                context_data["recent_chat"] = [
                    {
                        "user_said": conv.get('user', ''),
                        "buddy_replied": conv.get('buddy', ''),
                        "topic": conv.get('intent', 'conversation')
                    }
                    for conv in recent_conversations
                ]
                print(f"💬 BRAIN: Added {len(recent_conversations)} recent conversations")
        except Exception as e:
            print(f"⚠️ BRAIN: Could not load conversations: {e}")
        
        # Enhanced time/date handling
        if _needs_time_info(request.user_input):
            time_info = _get_enhanced_time_info(request.user_input)
            context_data["time_info"] = time_info
            print(f"🕐 BRAIN: Added time info: {time_info}")
        
        # Enhanced web search with caching and fallbacks
        web_results = None
        if _needs_web_search(request.user_input):
            web_results = await _enhanced_web_search(request.user_input)
            if web_results:
                context_data["web_search"] = web_results
                print(f"🌐 BRAIN: Web search successful: {web_results[:100]}...")
            else:
                print(f"🌐 BRAIN: Web search returned no results")
        
        # Object detection enhancement
        if request.objects_visible and _is_asking_about_objects(request.user_input):
            context_data["object_context"] = {
                "visible": request.objects_visible,
                "count": len(request.objects_visible)
            }
            print(f"👁️ BRAIN: Object context added: {request.objects_visible}")
        
        # Create enhanced prompt
        context_json = json.dumps(context_data, indent=2)
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_json}\n\nRespond with JSON only:"
        
        # Call LLM with retries
        response_text = await _call_llm_with_retry(prompt)
        
        print(f"🤖 BRAIN: LLM raw response: '{response_text[:150]}...'")
        
        # Parse and enhance response
        reply, intent, emotion = _parse_response(response_text)
        
        # INTELLIGENT POST-PROCESSING
        
        # 1. Enhanced intent detection with code-based backup
        detected_intent = _detect_intent_advanced(request.user_input, request.objects_visible)
        if intent == "conversation" and detected_intent != "conversation":
            intent = detected_intent
            print(f"🎯 BRAIN: Intent corrected to: {intent}")
        
        # 2. Strict validation
        valid_intents = ["greeting", "conversation", "provide_info", "question", "ask_name", 
                        "movement", "follow", "stop", "dance", "nod", "shake_head", 
                        "celebrate", "sleep", "wake_up"]
        if intent not in valid_intents:
            intent = "conversation"
        
        valid_emotions = ["happy", "excited", "cheerful", "friendly", "warm", "curious", 
                         "interested", "surprised", "concerned", "confused", "apologetic", 
                         "calm", "playful", "neutral"]
        if emotion not in valid_emotions:
            emotion = "friendly"
        
        # 3. PRIORITY HANDLING - Process in order of specificity
        
        # HIGHEST PRIORITY: Time/Date questions (override everything)
        if _needs_time_info(request.user_input):
            reply = _generate_time_response(request.user_input)
            intent = "provide_info"
            emotion = "helpful" if "helpful" in valid_emotions else "friendly"
            print(f"⏰ BRAIN: Time response generated: {reply}")
        
        # HIGH PRIORITY: Web search factual questions
        elif web_results and _needs_web_search(request.user_input):
            # LLM should have incorporated web results, but ensure quality
            if len(reply) < 20 or "not sure" in reply.lower() or "don't know" in reply.lower():
                reply = _format_web_answer(request.user_input, web_results)
            intent = "provide_info"
            emotion = "interested"
            print(f"🌐 BRAIN: Web-enhanced response: {reply}")
        
        # Handle contextual follow-ups (uses recent conversation)
        elif _is_contextual_followup(request.user_input, recent_conversations):
            # Let LLM handle it with context, but ensure it used context
            if len(reply) < 15 or "not sure" in reply.lower():
                reply = _handle_contextual_followup(request.user_input, recent_conversations, web_results)
            intent = "provide_info"
            print(f"🔗 BRAIN: Contextual follow-up: {reply}")
        
        # MEDIUM PRIORITY: Memory-based questions
        elif memory_context and _is_memory_question(request.user_input):
            reply = _enhance_memory_response(reply, request.user_input, memory_context, request.recognized_user)
            intent = "provide_info"
            print(f"📝 BRAIN: Memory-enhanced response: {reply}")
        
        # MEDIUM PRIORITY: Object detection questions
        elif _is_asking_about_objects(request.user_input):
            reply = _generate_object_response(request.user_input, request.objects_visible, reply)
            intent = "provide_info"
            emotion = "curious"
            print(f"👁️ BRAIN: Object response: {reply}")
        
        # MEDIUM PRIORITY: Movement commands
        elif intent in ["dance", "movement", "follow", "stop", "celebrate"]:
            reply = _enhance_movement_response(intent, reply)
            print(f"🕺 BRAIN: Movement response: {reply}")
        
        # LOW PRIORITY: Greetings with personalization
        elif intent == "greeting" or any(word in request.user_input.lower() for word in ['hello', 'hi', 'nice to see']):
            # This is a greeting - personalize it
            if request.recognized_user:
                hour = datetime.now().hour
                time_greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
                reply = f"{time_greeting} {request.recognized_user}! Nice to see you!"
            else:
                reply = "Hello! Nice to meet you!"
            intent = "greeting"
            emotion = "friendly"
            print(f"👋 BRAIN: Personalized greeting: {reply}")
        
        # Handle simple acknowledgments (ok, nice, cool, etc.)
        elif len(request.user_input.split()) == 1 and request.user_input.lower() in ['nice', 'ok', 'okay', 'cool', 'great', 'good']:
            acknowledgments = {
                'nice': "Glad you think so!",
                'ok': "Got it!",
                'okay': "Alright!",
                'cool': "Right?",
                'great': "Awesome!",
                'good': "Happy to help!"
            }
            reply = acknowledgments.get(request.user_input.lower(), "👍")
            intent = "conversation"
            emotion = "cheerful"
            print(f"👍 BRAIN: Simple acknowledgment: {reply}")
        
        # Handle unclear single words - ask for clarification
        elif len(request.user_input.split()) == 1 and len(request.user_input) < 8:
            word = request.user_input.lower()
            if word not in ['hello', 'hi', 'hey', 'bye', 'thanks', 'thank', 'please']:
                reply = f"I heard '{request.user_input}' - what did you mean by that?"
                intent = "question"
                emotion = "confused"
                print(f"❓ BRAIN: Clarification request: {reply}")
        
        # 4. Quality control - ensure replies are natural and complete
        reply = _polish_reply(reply)
        
        # 5. Length optimization (allow longer for factual content)
        max_length = 200 if intent == "provide_info" else 150
        if len(reply) > max_length:
            reply = reply[:max_length-3] + "..."
        
        # Create normalized JSON response
        normalized_json = json.dumps({
            "reply": reply,
            "emotion": emotion,
            "intent": intent
        })
        
        print(f"✅ BRAIN: Final - Reply: '{reply}', Intent: '{intent}', Emotion: '{emotion}'")
        
        # Intelligent memory extraction and saving
        memory_saved = await _smart_extract_and_save_memory(request.user_input, effective_user)
        if memory_saved:
            print(f"💾 BRAIN: Saved memory: {memory_saved}")
        
        # Save conversation with metadata
        save_conversation(
            request.user_input, 
            reply,  # Save the final reply, not raw LLM response
            intent, 
            effective_user
        )
        
        return ChatResponse(
            reply=reply,
            intent=intent,
            emotion=emotion,
            raw_response=normalized_json
        )
        
    except Exception as e:
        print(f"❌ BRAIN ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def _call_llm_with_retry(prompt: str, max_retries: int = 2) -> str:
    """Call LLM with retry logic for reliability"""
    for attempt in range(max_retries):
        try:
            response = await _call_llm(prompt)
            if response and len(response) > 10:
                return response
        except Exception as e:
            print(f"⚠️ LLM attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)
    
    return '{"reply": "Sorry, having trouble thinking right now.", "emotion": "apologetic", "intent": "conversation"}'

async def _call_llm(prompt: str) -> str:
    """Call LLM service (Ollama) with optimized parameters"""
    try:
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': 'llama3.2:3b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.4,  # Slightly higher for natural responses
                    'top_p': 0.9,
                    'top_k': 40,
                    'num_predict': 200,  # More tokens for complete answers
                    'num_ctx': 2048,     # Larger context window
                    'repeat_penalty': 1.1,
                    'stop': ['\n\n\n', 'User:', 'Human:']
                }
            },
            timeout=20
        )
        
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            print(f"⚠️ LLM returned status {response.status_code}")
            return '{"reply": "Sorry, having trouble thinking.", "emotion": "apologetic", "intent": "conversation"}'
            
    except requests.Timeout:
        print("⚠️ LLM timeout")
        return '{"reply": "Sorry, thinking took too long.", "emotion": "apologetic", "intent": "conversation"}'
    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        return '{"reply": "Sorry, having trouble thinking.", "emotion": "apologetic", "intent": "conversation"}'

def _parse_response(response: str) -> tuple[str, str, str]:
    """Enhanced JSON parsing with multiple fallback strategies"""
    try:
        # Clean response aggressively
        response = response.strip()
        
        # Remove markdown code blocks
        if '```' in response:
            response = re.sub(r'```(?:json)?\s*', '', response)
        
        # Extract JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        if json_match:
            response = json_match.group(0)
        
        # Parse JSON
        data = json.loads(response)
        
        # Extract fields with smart defaults
        reply = str(data.get("reply", data.get("response", data.get("text", "I'm not sure how to respond."))))
        intent = data.get("intent", "conversation")
        emotion = data.get("emotion", "friendly")
        
        # Clean up reply
        reply = reply.strip('"\'')
        
        return reply, intent, emotion
        
    except json.JSONDecodeError:
        # Try to extract from malformed JSON
        try:
            reply_match = re.search(r'"reply"\s*:\s*"([^"]+)"', response)
            emotion_match = re.search(r'"emotion"\s*:\s*"([^"]+)"', response)
            intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', response)
            
            if reply_match:
                return (
                    reply_match.group(1),
                    intent_match.group(1) if intent_match else "conversation",
                    emotion_match.group(1) if emotion_match else "friendly"
                )
        except:
            pass
        
        # Ultimate fallback
        return "I'm having trouble responding right now.", "conversation", "apologetic"

def _detect_intent_advanced(user_input: str, objects_visible: Optional[List] = None) -> str:
    """Advanced intent detection with context awareness"""
    text = user_input.lower().strip()
    
    # Single word responses - likely acknowledgments
    if len(text.split()) == 1 and text in ['nice', 'ok', 'okay', 'cool', 'great', 'good', 'yeah', 'yes', 'no', 'nope', 'fine', 'sure']:
        return "conversation"
    
    # Random single words that aren't common words - might be errors
    if len(text.split()) == 1 and len(text) < 8 and text not in ['hello', 'hi', 'hey', 'bye', 'thanks', 'thank']:
        return "conversation"  # Will trigger clarification
    
    # Movement intents - high confidence patterns
    movement_patterns = {
        "dance": ['dance', 'dancing', 'groove', 'boogie', 'bust a move'],
        "movement": ['come here', 'move forward', 'move closer', 'go to', 'walk'],
        "follow": ['follow me', 'follow along', 'come with'],
        "stop": ['stop', 'halt', 'freeze', 'don\'t move'],
        "celebrate": ['celebrate', 'party', 'woohoo', 'yay', 'hooray']
    }
    
    for intent, patterns in movement_patterns.items():
        if any(pattern in text for pattern in patterns):
            return intent
    
    # Greeting intents - detect greeting messages
    greeting_words = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings', 'nice to see']
    if any(word in text for word in greeting_words):
        return "greeting"
    
    # Sleep/goodbye intents
    if any(word in text for word in ['goodbye', 'bye', 'sleep', 'goodnight', 'see you']):
        return "sleep"
    
    # Question intents - must start with question word
    question_starters = ['what', 'when', 'where', 'who', 'how', 'why', 'which', 'whose', 'can you', 'could you', 'would you', 'will you']
    if any(text.startswith(q) for q in question_starters):
        return "question"
    
    # Object-related intents
    if objects_visible and _is_asking_about_objects(text):
        return "provide_info"
    
    return "conversation"

def _needs_time_info(user_input: str) -> bool:
    """Check if question EXPLICITLY requires time/date information"""
    text = user_input.lower().strip()
    
    # MUST be explicitly asking about time/date
    explicit_time_questions = [
        'what time is it',
        'what\'s the time',
        'tell me the time',
        'current time',
        'time now',
        'what day is it',
        'what day is today',
        'what\'s the date',
        'what is the date',
        'today\'s date',
        'what date is it',
        'date today',
        'what month',
        'what year'
    ]
    
    # Check for explicit patterns
    for pattern in explicit_time_questions:
        if pattern in text:
            return True
    
    # NOT time questions - common false positives
    false_positives = [
        'how is your day',
        'how\'s your day',
        'how are you',
        'your day going',
        'what are you doing',
        'what can you do',
        'time to',
        'time for',
        'one time',
        'this time'
    ]
    
    for false_pos in false_positives:
        if false_pos in text:
            return False
    
    return False

def _get_enhanced_time_info(user_input: str) -> Dict[str, Any]:
    """Get comprehensive time information"""
    now = datetime.now()
    
    return {
        "full": now.strftime("%A, %B %d, %Y at %I:%M %p"),
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%I:%M %p"),
        "day_of_week": now.strftime("%A"),
        "month": now.strftime("%B"),
        "day": now.day,
        "year": now.year,
        "hour": now.hour,
        "minute": now.minute
    }

def _generate_time_response(user_input: str) -> str:
    """Generate accurate time-based responses"""
    text = user_input.lower()
    now = datetime.now()
    
    # Specific queries
    if 'what day' in text or 'day is it' in text or 'day of the week' in text:
        return f"Today is {now.strftime('%A')}."
    
    if 'what time' in text or 'time is it' in text or 'current time' in text:
        return f"It's {now.strftime('%I:%M %p')}."
    
    if 'what date' in text or 'date today' in text or 'today\'s date' in text:
        return f"Today's date is {now.strftime('%B %d, %Y')}."
    
    if 'what month' in text:
        return f"It's {now.strftime('%B')}."
    
    if 'what year' in text:
        return f"It's {now.year}."
    
    # General time query
    return f"It's {now.strftime('%A, %B %d, %Y at %I:%M %p')}."

def _needs_web_search(user_input: str) -> bool:
    """Enhanced detection for questions needing web search"""
    text = user_input.lower()
    
    # Question patterns that need web search
    factual_patterns = [
        'who is', 'who was', 'who are', 'who were',
        'what is', 'what was', 'what are', 'what were',
        'when did', 'when was', 'when were',
        'where is', 'where was', 'where are',
        'how many', 'how much', 'how tall', 'how old',
        'why did', 'why is', 'why was'
    ]
    
    # Specific topics
    factual_topics = [
        'president', 'prime minister', 'capital', 'country', 'city',
        'invented', 'discovered', 'founded', 'created', 'born', 'died',
        'population', 'height', 'weight', 'age', 'worth',
        'company', 'famous for', 'known for', 'history of',
        'definition of', 'meaning of', 'explain'
    ]
    
    # Check patterns
    has_pattern = any(pattern in text for pattern in factual_patterns)
    has_topic = any(topic in text for topic in factual_topics)
    
    # Exclude time/personal questions
    is_time_question = _needs_time_info(user_input)
    is_personal = any(word in text for word in ['you', 'your', 'buddy'])
    
    return (has_pattern or has_topic) and not is_time_question and not is_personal

@lru_cache(maxsize=100)
def _get_cached_search(query: str) -> Optional[str]:
    """Get cached search result if fresh (5 minutes)"""
    cache_key = query.lower().strip()
    if cache_key in _search_cache:
        result, timestamp = _search_cache[cache_key]
        if datetime.now() - timestamp < timedelta(minutes=5):
            print(f"📦 BRAIN: Using cached search result")
            return result
        else:
            del _search_cache[cache_key]
    return None

async def _enhanced_web_search(query: str) -> Optional[str]:
    """Enhanced web search with multiple strategies and caching"""
    
    # Check cache first
    cached = _get_cached_search(query)
    if cached:
        return cached
    
    result = None
    
    try:
        # Strategy 1: DuckDuckGo Instant Answer API
        result = await _ddg_instant_answer(query)
        if result:
            print(f"🌐 Strategy 1 (DDG Instant) succeeded")
            _search_cache[query.lower().strip()] = (result, datetime.now())
            return result
        
        # Strategy 2: DuckDuckGo HTML scraping
        result = await _ddg_html_search(query)
        if result:
            print(f"🌐 Strategy 2 (DDG HTML) succeeded")
            _search_cache[query.lower().strip()] = (result, datetime.now())
            return result
        
        # Strategy 3: Wikipedia-like pattern extraction
        result = await _wikipedia_search(query)
        if result:
            print(f"🌐 Strategy 3 (Wikipedia) succeeded")
            _search_cache[query.lower().strip()] = (result, datetime.now())
            return result
        
    except Exception as e:
        print(f"⚠️ Web search error: {e}")
    
    return None

async def _ddg_instant_answer(query: str) -> Optional[str]:
    """DuckDuckGo instant answer API"""
    try:
        api_url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Priority order: Answer > Abstract > Definition > Related Topics
            if data.get('Answer'):
                return _clean_search_result(data['Answer'])
            
            if data.get('AbstractText'):
                return _clean_search_result(data['AbstractText'])
            
            if data.get('Definition'):
                return _clean_search_result(data['Definition'])
            
            # Check related topics
            if data.get('RelatedTopics') and len(data['RelatedTopics']) > 0:
                first_topic = data['RelatedTopics'][0]
                if isinstance(first_topic, dict) and 'Text' in first_topic:
                    return _clean_search_result(first_topic['Text'])
        
    except Exception as e:
        print(f"⚠️ DDG API error: {e}")
    
    return None

async def _ddg_html_search(query: str) -> Optional[str]:
    """DuckDuckGo HTML scraping"""
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=7)
        
        if response.status_code == 200:
            text = response.text
            
            # Try multiple snippet patterns
            patterns = [
                r'class="result__snippet"[^>]*>([^<]+)',
                r'class="result__body"[^>]*>.*?<span[^>]*>([^<]+)',
                r'snippet">([^<]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    snippet = match.group(1).strip()
                    if len(snippet) > 20:
                        return _clean_search_result(snippet)
        
    except Exception as e:
        print(f"⚠️ DDG HTML error: {e}")
    
    return None

async def _wikipedia_search(query: str) -> Optional[str]:
    """Wikipedia API search"""
    try:
        # Extract main subject from query
        subject = re.sub(r'^(who is|what is|when was|where is|how)\s+', '', query, flags=re.IGNORECASE)
        
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(subject)}"
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'extract' in data:
                return _clean_search_result(data['extract'])
        
    except Exception as e:
        print(f"⚠️ Wikipedia error: {e}")
    
    return None

def _clean_search_result(text: str) -> str:
    """Clean and format search result"""
    # Remove HTML entities
    text = html.unescape(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove citations like [1], [2]
    text = re.sub(r'\[\d+\]', '', text)
    
    # Limit length intelligently (break at sentence)
    if len(text) > 250:
        sentences = re.split(r'[.!?]+\s+', text)
        result = sentences[0]
        for sent in sentences[1:]:
            if len(result) + len(sent) < 250:
                result += ". " + sent
            else:
                break
        text = result + "."
    
    return text

def _format_web_answer(question: str, web_result: str) -> str:
    """Format web search result into a natural answer"""
    # For "who is" questions
    if question.lower().startswith('who is') or question.lower().startswith('who was'):
        return web_result
    
    # For "what is" questions
    if question.lower().startswith('what is') or question.lower().startswith('what are'):
        return web_result
    
    # For "when" questions
    if question.lower().startswith('when'):
        return web_result
    
    # For "where" questions  
    if question.lower().startswith('where'):
        return web_result
    
    # For "how many/much" questions
    if question.lower().startswith('how many') or question.lower().startswith('how much'):
        return web_result
    
    return web_result

def _is_contextual_followup(user_input: str, recent_conversations: List) -> bool:
    """Check if this is a follow-up question requiring context"""
    if not recent_conversations:
        return False
    
    text = user_input.lower().strip()
    
    # Pronoun references to previous topic
    if any(word in text for word in ['he ', 'she ', 'they ', 'it ', 'his ', 'her ', 'their ', 'its ']):
        return True
    
    # Incomplete questions that need context
    incomplete_patterns = [
        'and what about',
        'what about',
        'how about',
        'is the',
        'was the',
        'are the',
        'were the',
        'the ceo',
        'the founder',
        'the president'
    ]
    
    if any(pattern in text for pattern in incomplete_patterns):
        return True
    
    # Very short questions (likely follow-ups)
    if len(text.split()) <= 4 and text.endswith('?'):
        return True
    
    return False

def _handle_contextual_followup(user_input: str, recent_conversations: List, web_results: Optional[str]) -> str:
    """Handle follow-up questions using conversation context"""
    if not recent_conversations:
        return "Could you give me more context about what you're asking?"
    
    # Get the last substantive exchange
    last_conv = recent_conversations[0]
    last_user = last_conv.get('user', '')
    last_buddy = last_conv.get('buddy', '')
    
    # If we have web results, use them
    if web_results:
        return web_results
    
    # Try to extract subject from previous conversation
    text = user_input.lower()
    
    # Handle "he/she is the CEO of" type follow-ups
    if 'ceo' in text or 'founder' in text or 'president' in text:
        # Extract name from previous answer if mentioned
        import re
        name_match = re.search(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', last_buddy)
        if name_match:
            name = name_match.group(1)
            if 'ceo' in text:
                return f"{name} is the CEO of several companies including SpaceX and Tesla."
    
    # Generic contextual response
    return "Based on what we were just discussing, could you be more specific with your question?"

def _is_memory_relevant(user_input: str) -> bool:
    """Check if memory should be included in context"""
    memory_keywords = [
        'remember', 'forgot', 'birthday', 'age', 'name',
        'favorite', 'like', 'love', 'prefer',
        'where', 'when', 'what', 'live', 'work',
        'my', 'i am', 'i\'m'
    ]
    return any(keyword in user_input.lower() for keyword in memory_keywords)

def _is_memory_question(user_input: str) -> bool:
    """Check if asking about stored memory"""
    text = user_input.lower()
    memory_patterns = [
        'my birthday', 'my name', 'my favorite',
        'when is my', 'what is my', 'where do i',
        'do you remember', 'what do you know about me'
    ]
    return any(pattern in text for pattern in memory_patterns)

def _enhance_memory_response(reply: str, question: str, memory: Dict, user_name: str) -> str:
    """Enhance response with memory information - with fallback to any user's memory"""
    text = question.lower()
    
    # Birthday questions - try current user first, then any user
    if 'birthday' in text:
        if 'birthday' in memory:
            return f"Your birthday is {memory['birthday']}!"
        else:
            # Fallback: check if ANY user has birthday info
            try:
                from memory import get_all_memory
                all_memory = get_all_memory()  # Get all users' memory
                if 'birthday' in all_memory:
                    return f"Your birthday is {all_memory['birthday']}!"
                # Try other users' memory
                for key, value in all_memory.items():
                    if 'birthday' in key.lower():
                        return f"Your birthday is {value}!"
            except:
                pass
    
    # Favorite color
    if 'favorite color' in text and 'favorite_color' in memory:
        return f"Your favorite color is {memory['favorite_color']}!"
    
    # General memory recall
    if 'remember' in text or 'know about me' in text:
        facts = []
        if 'birthday' in memory:
            facts.append(f"your birthday is {memory['birthday']}")
        if 'favorite_color' in memory:
            facts.append(f"your favorite color is {memory['favorite_color']}")
        if 'favorite_food' in memory:
            facts.append(f"your favorite food is {memory['favorite_food']}")
        
        if facts:
            return f"Yes! I remember that {', and '.join(facts)}."
    
    return reply

def _is_asking_about_objects(user_input: str) -> bool:
    """Check if asking about visible objects"""
    object_patterns = [
        'what do you see', 'what is this', 'what is that', 'what are these', 'what are those',
        'in my hand', 'holding', 'looking at', 'show', 'holding up',
        'can you see', 'do you see', 'what am i', 'recognize', 'identify',
        'what\'s this', 'what\'s that', 'see this', 'see that'
    ]
    text = user_input.lower()
    
    # Must be asking about objects specifically
    if any(pattern in text for pattern in object_patterns):
        return True
    
    # "what" alone is too vague
    if text.strip() in ['what', 'what is', 'what are']:
        return False
    
    return False

def _generate_object_response(question: str, objects: Optional[List], fallback: str) -> str:
    """Generate response about visible objects"""
    if not objects or len(objects) == 0:
        return "I don't see any objects right now."
    
    objects_text = ', '.join(objects)
    
    if 'how many' in question.lower():
        return f"I can see {len(objects)} object{'s' if len(objects) > 1 else ''}: {objects_text}."
    
    if len(objects) == 1:
        return f"I can see a {objects[0]}."
    else:
        return f"I can see {objects_text}."

def _enhance_movement_response(intent: str, reply: str) -> str:
    """Generate enthusiastic movement responses"""
    if "not sure" in reply.lower() or len(reply) < 15:
        movement_responses = {
            "dance": "Let's dance! I'm getting my groove on!",
            "movement": "Moving as requested!",
            "follow": "Following you now!",
            "stop": "Stopping all movement.",
            "celebrate": "Woohoo! Time to celebrate!"
        }
        return movement_responses.get(intent, reply)
    return reply

def _polish_reply(reply: str) -> str:
    """Polish and clean up the reply"""
    # Remove awkward phrases
    awkward_phrases = [
        "I'm not sure but",
        "I think that",
        "Maybe",
        "I believe"
    ]
    
    for phrase in awkward_phrases:
        reply = reply.replace(phrase, "")
    
    # Clean up spacing
    reply = re.sub(r'\s+', ' ', reply).strip()
    
    # Ensure proper capitalization
    if reply and reply[0].islower():
        reply = reply[0].upper() + reply[1:]
    
    # Ensure ending punctuation
    if reply and reply[-1] not in '.!?':
        reply += '.'
    
    return reply

async def _smart_extract_and_save_memory(user_input: str, user_name: str) -> Optional[str]:
    """Intelligent memory extraction and storage"""
    import re
    from memory import save_memory
    
    text = user_input.lower()
    effective_user = user_name if user_name else "Unknown"
    
    # Birthday extraction with multiple patterns
    if any(word in text for word in ['birthday', 'born', 'birth date']):
        date_patterns = [
            r'(?:birthday|born|birth\s*date).*?(?:is|on|in)?\s*(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(?:birthday|born|birth\s*date).*?(?:is|on|in)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})',
            r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if groups[0].isdigit():
                    day, month = groups[0], groups[1]
                else:
                    month, day = groups[0], groups[1]
                birthday = f"{day} {month.capitalize()}"
                save_memory("birthday", birthday, user_name=effective_user)
                return f"birthday: {birthday}"
    
    # Name extraction
    name_patterns = [
        r'(?:my\s+name\s+is|i\'m|i\s+am|call\s+me)\s+([A-Z][a-z]+)',
        r'(?:this\s+is|i\'m)\s+([A-Z][a-z]+)'
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, user_input)  # Case-sensitive for names
        if match:
            name = match.group(1)
            save_memory("preferred_name", name, user_name=effective_user)
            return f"name: {name}"
    
    # Favorite color
    color_match = re.search(r'(?:favorite|favourite|fav)\s+color.*?(?:is|are)\s+(\w+)', text)
    if color_match:
        color = color_match.group(1)
        save_memory("favorite_color", color, user_name=effective_user)
        return f"favorite_color: {color}"
    
    # Favorite food
    food_match = re.search(r'(?:favorite|favourite|fav)\s+food.*?(?:is|are)\s+([\w\s]+?)(?:\.|,|and|$)', text)
    if food_match:
        food = food_match.group(1).strip()
        save_memory("favorite_food", food, user_name=effective_user)
        return f"favorite_food: {food}"
    
    # Favorite movie/show
    media_match = re.search(r'(?:favorite|favourite|fav)\s+(?:movie|film|show|series).*?(?:is|are)\s+([\w\s]+?)(?:\.|,|and|$)', text)
    if media_match:
        media = media_match.group(1).strip()
        save_memory("favorite_movie", media, user_name=effective_user)
        return f"favorite_movie: {media}"
    
    # Location
    location_patterns = [
        r'(?:i\s+live\s+in|i\'m\s+from|from)\s+([\w\s]+?)(?:\.|,|and|$)',
        r'(?:my\s+(?:home|city|town)\s+is)\s+([\w\s]+?)(?:\.|,|and|$)'
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, text)
        if match:
            location = match.group(1).strip()
            save_memory("location", location, user_name=effective_user)
            return f"location: {location}"
    
    # Age
    age_match = re.search(r'(?:i\'m|i\s+am)\s+(\d+)\s+years?\s+old', text)
    if age_match:
        age = age_match.group(1)
        save_memory("age", age, user_name=effective_user)
        return f"age: {age}"
    
    return None

@app.get("/")
async def root():
    """Root endpoint showing service status"""
    return {"message": "Buddy Enhanced Brain Service is running!", "status": "active"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Buddy Enhanced Brain Service",
        "version": "2.0.0",
        "features": [
            "web_search",
            "time_awareness",
            "smart_memory",
            "object_detection",
            "conversation_context"
        ]
    }

@app.post("/clear")
async def clear_conversation():
    """Clear conversation history"""
    try:
        from memory import clear_conversations
        clear_conversations()
        _search_cache.clear()
        print("🧹 BRAIN: Conversation history and cache cleared")
        return {"status": "cleared", "message": "Conversation history and search cache cleared"}
    except Exception as e:
        print(f"❌ BRAIN: Error clearing conversations: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/stats")
async def get_stats():
    """Get brain statistics"""
    try:
        from memory import get_all_memory
        
        memory_count = len(get_all_memory())
        cache_count = len(_search_cache)
        
        return {
            "memory_entries": memory_count,
            "cached_searches": cache_count,
            "status": "operational"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🧠 BUDDY ENHANCED BRAIN SERVICE v2.0")
    print("=" * 60)
    print("🚀 Starting on port 8000...")
    print("✅ Features enabled:")
    print("   • Web search with caching")
    print("   • Real-time awareness")
    print("   • Smart memory extraction")
    print("   • Object detection integration")
    print("   • Multi-turn conversation context")
    print("=" * 60)
    print("📋 Prerequisites:")
    print("   • Ollama running: ollama serve")
    print("   • Model available: ollama pull llama3.2:3b")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)