"""
openrouter_client.py

Alfred 2.0 - shared OpenRouter access

A single place for headers, timeouts, retries and error handling.
Previously this logic sat directly in the web client; now the answer, the
memory filter and the memory extractor can all share the same code.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

try:
    from config import (
        OPENROUTER_API_KEY,
        OPENROUTER_URL,
        OPENROUTER_REFERER,
        OPENROUTER_TITLE,
    )
except Exception:  # fallback, so the module stays importable standalone as well
    import os
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_REFERER = "http://localhost:8505"
    OPENROUTER_TITLE = "Alfred Assistant"


class OpenRouterError(RuntimeError):
    """Error while calling OpenRouter (network, HTTP status, response format)."""


def build_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    key = (api_key or OPENROUTER_API_KEY or "").strip()

    if not key:
        raise OpenRouterError(
            "No OpenRouter API key found. Set the environment variable "
            "OPENROUTER_API_KEY (see the comment in config.py)."
        )

    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": OPENROUTER_TITLE,
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    }


def chat_completion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 60,
    retries: int = 2,
    retry_backoff: float = 1.5,
) -> Dict[str, Any]:
    """Calls OpenRouter and returns the parsed JSON response.

    Adds the extra key "_elapsed" (seconds) to the response, so the
    caller can display the duration without timing it themselves.

    Raises OpenRouterError if no valid result is available after all
    attempts. On 4xx errors (except 429) there is NO retry - a wrong
    model name or an invalid key does not get any better on the second
    attempt.
    """

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "usage": {"include": True},
    }

    if tools:
        payload["tools"] = tools
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = build_headers(api_key)
    last_error: Optional[str] = None
    started = time.perf_counter()

    for attempt in range(1, retries + 1):
        try:
            res = requests.post(
                url=OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {e}"
            if attempt < retries:
                time.sleep(retry_backoff * attempt)
            continue

        if res.status_code == 200:
            try:
                data = res.json()
            except ValueError as e:
                last_error = f"Response was not valid JSON: {e}"
                if attempt < retries:
                    time.sleep(retry_backoff * attempt)
                continue

            data["_elapsed"] = time.perf_counter() - started
            return data

        body = res.text[:300]
        last_error = f"HTTP {res.status_code}: {body}"

        # 4xx other than 429 is an error on our side -> no retry.
        if 400 <= res.status_code < 500 and res.status_code != 429:
            break

        if attempt < retries:
            time.sleep(retry_backoff * attempt)

    raise OpenRouterError(f"OpenRouter call failed ({model}): {last_error}")


def extract_text(data: Dict[str, Any]) -> str:
    """Fetches the response text out of an OpenRouter response."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise OpenRouterError(f"Unexpected response format: {e}")

    content = message.get("content")

    if isinstance(content, str):
        return content

    # Some models/providers return content as a list of blocks.
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)

    return ""


def extract_usage(data: Dict[str, Any]) -> Dict[str, Any]:
    """Returns token counts, actual cost and the number of web searches.

    OpenRouter reports the real cost in USD along with the response when
    'usage': {'include': True} is set in the payload - which is why we do
    not have to calculate anything ourselves here.
    """
    usage = data.get("usage") or {}

    search_count = 0
    details = usage.get("server_tool_use_details")
    if isinstance(details, dict):
        search_count = details.get("web_search_requests", 0) or 0

    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cost": usage.get("cost"),
        "web_search_requests": search_count,
        "elapsed": data.get("_elapsed"),
    }
