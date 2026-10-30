import json
import re
import os
from .ai_agent import ask_llm

def generate_tests_for_kernel(kernel_name: str, signature: str, out_file: str):
    prompt = f"""
    You are a QA engineer testing a GPU math library in Python (using numpy).
    Kernel: {kernel_name}
    Signature: {signature}
    
    Write 2 edge-case pytest tests for this kernel. Output JSON only in this format:
    {{"tests": [{{"name": "test_name", "code": "def test_name():\\n    pass"}}]}}
    """
    
    print(f"Asking AI for tests for {kernel_name}...")
    response = ask_llm(prompt)
    
    # Extract JSON from possible markdown blocks
    match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
    json_str = match.group(1) if match else response
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print("Error: LLM did not return valid JSON.")
        return
        
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write("import pytest\nimport numpy as np\nimport hipmath\n\n")
        for t in data.get("tests", []):
            f.write(f"# AI-generated test: {t['name']}\n")
            f.write(t["code"] + "\n\n")
            
    print(f"Wrote {len(data.get('tests', []))} tests to {out_file}")