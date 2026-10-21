"""
memory_manager_v2_1.py

Alfred Memory Manager v2.1

Improvements:
- Stronger system prompt
- Robust JSON extraction
- Automatic retry on invalid JSON
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import requests


class MemoryManager:

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: int = 120,
        debug: bool = False,
        max_retries: int = 2,
        keep_alive: str = "30m",
        num_thread: int | None = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.debug = debug
        self.max_retries = max_retries
        self.keep_alive = keep_alive
        self.num_thread = num_thread

        self.logger = logging.getLogger("MemoryManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def build_system_prompt(self) -> str:
        # The prompt is kept in English on purpose and needs no
        # translation: the model only ever answers with the ID numbers of
        # the selected memories ({"selected_ids":[...]}), never with text.
        # The selection is therefore language-independent by design and
        # works just as well on German memories.
        return (
            "You are Alfred's Memory Manager.\n"
            "Your ONLY task is selecting which stored memories (by their number) "
            "are relevant to the user's question.\n"
            "Never answer the user's question.\n"
            "Never summarize.\n"
            "Never invent memories.\n"
            "Never rewrite or repeat the memory text - only refer to memories by their number.\n"
            "IMPORTANT: There is NO fixed or expected number of memories to select. It could be "
            "zero, one, two, five, or every single memory in the list - it depends ENTIRELY on "
            "how many are actually relevant to the question. If the question relates to a topic "
            "covered by many stored memories (e.g. 'what trips have I taken' and there are 5 "
            "trips stored), select ALL of them, not just a couple.\n"
            "Prefer returning one relevant memory too many rather than omitting an important one.\n"
            "Return ONLY valid JSON matching this format (the numbers below are just an example "
            "of the FORMAT, not a hint about how many you should pick):\n"
            '{"selected_ids":[0,2,5,9]}\n'
            "If no memory is relevant, return: {\"selected_ids\":[]}"
        )

    def build_prompt(self, user_prompt: str, memories: List[str]) -> str:
        memory_text = "\n".join(f"{i}: {m}" for i, m in enumerate(memories))
        return f"""User question:
{user_prompt}

Stored memories (numbered):
{memory_text}
"""

    def call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        options = {"temperature": self.temperature}
        if self.num_thread:
            options["num_thread"] = self.num_thread

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": options,
            "keep_alive": self.keep_alive,
        }

        r = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def parse_response(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()

        if "```" in cleaned:
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON object found.")

        cleaned = cleaned[start:end + 1]

        data = json.loads(cleaned)

        if not isinstance(data.get("selected_ids"), list):
            raise ValueError("'selected_ids' missing or invalid.")

        return data

    def select_memories(self, user_prompt: str, memories: List[str]) -> List[str]:

        if not memories:
            return []

        system_prompt = self.build_system_prompt()
        prompt = self.build_prompt(user_prompt, memories)

        last_error = None

        for attempt in range(1, self.max_retries + 1):

            try:
                raw = self.call_ollama(system_prompt, prompt)
            except requests.exceptions.RequestException as e:
                last_error = e
                if self.debug:
                    self.logger.debug("Network error on attempt %s: %s", attempt, e)
                continue

            if self.debug:
                self.logger.debug("Attempt %s", attempt)
                self.logger.debug(raw)

            try:
                parsed = self.parse_response(raw)
                selected_ids = parsed["selected_ids"]

                # Map the IDs back onto the ORIGINAL texts instead of
                # trusting the model to repeat the text without mistakes.
                # Invalid/duplicate IDs are silently ignored.
                result = []
                seen_ids = set()
                for raw_id in selected_ids:
                    try:
                        idx = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    if idx in seen_ids or idx < 0 or idx >= len(memories):
                        continue
                    seen_ids.add(idx)
                    result.append(memories[idx])

                return result

            except Exception as ex:
                last_error = ex

                prompt += """

IMPORTANT:
Your previous answer was not valid JSON.

Return ONLY valid JSON.
No markdown.
No explanation.
"""

        raise RuntimeError(
            f"Memory selection failed after {self.max_retries} attempts: {last_error}"
        )
