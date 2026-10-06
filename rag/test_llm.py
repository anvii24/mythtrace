"""Smoke test: check that the LLM API key in .env works.

Loads LLM_API_KEY from .env (never printed) and sends one short prompt to Claude.
Run from the repo root:  python rag/test_llm.py
"""
import os
import sys

import anthropic
from dotenv import load_dotenv

MODEL = "claude-opus-5-5"


def main():
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        sys.exit("LLM_API_KEY not found in .env")

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=256,
            output_config={"effort": "low"},  # trivial prompt, no deep reasoning needed
            messages=[{"role": "user", "content": "Say hello in five words"}],
        )
    except anthropic.AuthenticationError:
        sys.exit("Authentication failed: check LLM_API_KEY in .env")
    except anthropic.APIError as e:
        sys.exit(f"API error: {e}")

    text = "".join(b.text for b in response.content if b.type == "text")
    print(f"Model: {response.model}")
    print(f"Reply: {text}")
    print(f"Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")


if __name__ == "__main__":
    main()
