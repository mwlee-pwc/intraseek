import requests
import json
import os
from typing import Iterator

def fetch_perplexity_answer(question: str) -> str:
    """
    Minimal Perplexity API helper.
    Requires env var PERPLEXITY_API_KEY. The exact endpoint and payload may need
    adjustment if Perplexity's API differs; this implementation uses a generic
    POST with a JSON body and Bearer auth. Errors are raised to be handled by caller.
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY not set")

    response = requests.post(
        'https://api.perplexity.ai/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'sonar-pro',
            'messages': [
                {
                    'role': 'user',
                    'content': question
                }
            ]
        }
    )

    # resp = requests.post(url, headers=headers, json=json, timeout=30)
    response.raise_for_status()
    try:
        response = response.json()["choices"][0]['message']['content']
    except Exception as e:
        response = "No Perplexity response"
    return response

def fetch_perplexity_stream(question: str) -> Iterator[str]:
    """
    Stream Perplexity response chunks. Yields text chunks as they arrive.
    NOTE: Adjust endpoint / parsing to match Perplexity API (SSE vs chunked JSON lines).
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY not set")

    url = 'https://api.perplexity.ai/chat/completions'  # adjust to real streaming endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        'model': 'sonar-pro',
        'messages': [
            {
                'role': 'user',
                'content': question
            }
        ],
        'stream': True,
    }

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()

            if line.startswith("data:"):
                line = line[len("data:"):].strip()

            if line == "[DONE]":
                break

            content_chunk = None
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue

            try:
                choices = obj.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    ch = choices[0]
                    delta = ch.get("delta") or {}
                    if isinstance(delta, dict):
                        content_chunk = delta.get("content") or (delta.get("message") or {}).get("content")
                    if not content_chunk and isinstance(ch.get("message"), dict):
                        content_chunk = ch["message"].get("content")
                    if not content_chunk:
                        content_chunk = ch.get("text") or ch.get("answer")
                if not content_chunk:
                    content_chunk = obj.get("text") or obj.get("answer")
            except Exception:
                content_chunk = None

            if content_chunk:
                yield content_chunk

if __name__ == "__main__":
    # Simple test
    question = "What is the capital of France?"
    try:
        answer = fetch_perplexity_answer(question)
        print(f"Q: {question}\nA: {answer}")
    except Exception as e:
        print(f"Error fetching answer: {e}")