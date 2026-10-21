"""
memory_filter_cloud.py

Alfred 2.0 - memory filter via OpenRouter

Same job as memory_manager_v2_1.py: pick the memories relevant to the
current question out of a numbered list of stored memories. The only
difference is that the selection runs on a cheap cloud model here
instead of on local Ollama.

Deliberately built as a subclass of MemoryManager: the system prompt,
JSON parsing, retry logic and above all the mapping of the IDs back onto
the ORIGINAL memory texts are thereby guaranteed to be identical. The
only thing swapped out is WHO answers the prompt.

Interface identical to MemoryManager:

    filter = CloudMemoryFilter(model="google/gemini-2.5-flash-lite")
    relevant = filter.select_memories(user_prompt="...", memories=[...])
"""

from __future__ import annotations

from typing import List, Optional

import requests

from memory_manager_v2_1 import MemoryManager
from openrouter_client import (
    OpenRouterError,
    chat_completion,
    extract_text,
    extract_usage,
)

try:
    from config import (
        MEMORY_FILTER_MODEL,
        MEMORY_FILTER_TIMEOUT,
        OPENROUTER_API_KEY,
    )
except Exception:
    MEMORY_FILTER_MODEL = "google/gemini-2.5-flash-lite"
    MEMORY_FILTER_TIMEOUT = 30
    OPENROUTER_API_KEY = ""


class CloudMemoryFilter(MemoryManager):

    def __init__(
        self,
        model: str = MEMORY_FILTER_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = MEMORY_FILTER_TIMEOUT,
        debug: bool = False,
        max_retries: int = 2,
    ):
        # The Ollama-specific parameters of the base class (host,
        # keep_alive, num_thread) are meaningless here and stay at their
        # defaults.
        super().__init__(
            model=model,
            temperature=temperature,
            timeout=timeout,
            debug=debug,
            max_retries=max_retries,
        )
        self.api_key = api_key or OPENROUTER_API_KEY
        self.last_usage = None

    def call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Name inherited from the base class - here the call of course
        goes to OpenRouter, not to Ollama. The base class's
        select_memories() calls exactly this method, which is why we keep
        the signature."""

        try:
            data = chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=self.api_key,
                temperature=self.temperature,
                timeout=self.timeout,
                retries=1,  # retries are controlled by the base class's select_memories()
            )
        except OpenRouterError as e:
            # The base class's select_memories() treats RequestException as
            # a network error (retry without the "was not valid JSON" extra).
            raise requests.exceptions.RequestException(str(e)) from e

        self.last_usage = extract_usage(data)

        return extract_text(data)


if __name__ == "__main__":

    demo_memories = [
        "01.01.2026: Jonas works as a Java developer.",
        "01.01.2026: Jonas works at Stadtwerke Musterstadt.",
        "02.01.2026: Jonas likes pizza.",
        "03.01.2026: Jonas lives with his family in Musterstadt.",
        "04.01.2026: Jonas is interested in astrophotography.",
    ]

    memory_filter = CloudMemoryFilter(debug=True)

    while True:
        frage = input("\nQuestion (empty quits): ")
        if not frage:
            break

        try:
            treffer = memory_filter.select_memories(frage, demo_memories)
        except (OpenRouterError, RuntimeError) as e:
            print(f"Error: {e}")
            continue

        print(f"\n{len(treffer)} of {len(demo_memories)} memories selected:")
        for t in treffer:
            print(f"  - {t}")

        if memory_filter.last_usage:
            u = memory_filter.last_usage
            print(f"\n  {u['total_tokens']} tokens, {u['elapsed']:.2f}s, cost: {u['cost']}")
