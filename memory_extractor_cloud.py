"""
memory_extractor_cloud.py

Alfred 2.0 - Memory extractor via OpenRouter

Uses EXACTLY the same prompt and the same post-processing as
memory_extractor_v5.py (atomic memories, type check against
ALLOWED_TYPES, question filter, duplicate check) - it is only a
cloud model that answers instead of llama3.1:8b locally.

That is intentional: the regression test (43 cases) stays valid this
way, and you can measure both backends directly against each other
without fiddling with the prompt.

Interface identical to MemoryExtractor:

    extractor = CloudMemoryExtractor()
    memories = extractor.extract("I work as a Java developer.")
"""

from __future__ import annotations

from typing import Optional

import requests

from memory_extractor_v5 import MemoryExtractor
from openrouter_client import OpenRouterError, chat_completion, extract_text, extract_usage

try:
    from config import (
        MEMORY_EXTRACTOR_CLOUD_MODEL,
        MEMORY_EXTRACTOR_TIMEOUT,
        OPENROUTER_API_KEY,
    )
except Exception:
    MEMORY_EXTRACTOR_CLOUD_MODEL = "google/gemini-2.5-flash-lite"
    MEMORY_EXTRACTOR_TIMEOUT = 60
    OPENROUTER_API_KEY = ""


class CloudMemoryExtractor(MemoryExtractor):

    def __init__(
        self,
        model: str = MEMORY_EXTRACTOR_CLOUD_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = MEMORY_EXTRACTOR_TIMEOUT,
        retries: int = 3,
        debug: bool = False,
    ):
        # host / keep_alive / num_thread of the base class are meaningless here.
        super().__init__(
            model=model,
            temperature=temperature,
            retries=retries,
            debug=debug,
        )
        self.api_key = api_key or OPENROUTER_API_KEY
        self.timeout = timeout
        self.last_usage = None
        self.backend_name = "OpenRouter"

    def _call(self, prompt: str) -> str:
        """Replaces the base class's Ollama call with OpenRouter.
        extract() of the base class calls exactly this method.

        OpenRouterError is deliberately translated into
        requests.RequestException: the retry loop in
        MemoryExtractor.extract() catches exactly this type and counts it
        as a network error. That way the base class stays unchanged and
        the warning "Ollama was unreachable on ALL attempts" also applies
        here in spirit."""

        try:
            data = chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                temperature=self.temperature,
                timeout=self.timeout,
                retries=1,  # retries are controlled by extract() of the base class
            )
        except OpenRouterError as e:
            raise requests.exceptions.RequestException(str(e)) from e

        self.last_usage = extract_usage(data)

        return extract_text(data)


if __name__ == "__main__":

    extractor = CloudMemoryExtractor(debug=False)

    while True:
        gespraech = input("\nConversation (empty to quit): ")
        if not gespraech:
            break

        try:
            memories = extractor.extract(gespraech)
        except OpenRouterError as e:
            print(f"Error: {e}")
            continue

        if not memories:
            print("No memories found.")
        else:
            print()
            for i, mem in enumerate(memories, 1):
                print(f"{i:02d}. [{mem['type']}] {mem['text']}")
                print(f"    Confidence: {mem['confidence']}  |  Reason: {mem['reason']}")

        if extractor.last_usage:
            u = extractor.last_usage
            print(f"\n  {u['total_tokens']} tokens, {u['elapsed']:.2f}s, cost: {u['cost']}")
