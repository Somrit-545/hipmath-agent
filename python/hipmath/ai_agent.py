import os
import json
import hashlib

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "ai_cache.json")

def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def ask_llm(prompt: str) -> str:
    """Query the LLM, using a local cache to avoid duplicate API calls."""
    cache = _load_cache()
    key = hashlib.sha256(prompt.encode()).hexdigest()
    
    if key in cache:
        return cache[key]

    api_key = os.environ.get("LLM_API_KEY")
    
    if api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
        )
        result = response.text
    else:
        # MOCK MODE: Returns dummy data so the pipeline works without an API key
        print("[ai_agent] No LLM_API_KEY found. Using mock response.")
        if "pytest" in prompt.lower():
            result = '```json\n{"tests": [{"name": "test_zero_length", "code": "def test_zero_length():\\n    assert hipmath.dot(np.array([]), np.array([])) == 0.0"}]}\n```'
        else:
            result = '```json\n{"suggestions": [{"title": "Use shared memory", "why": "Reduces global memory reads"}, {"title": "Vectorize loads", "why": "Improves memory bandwidth"}]}\n```'

    cache[key] = result
    _save_cache(cache)
    return result