"""
Local LLM client: Ollama (primary) and optional llama.cpp server.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct")
LLAMA_CPP_BASE_URL = os.environ.get("LLAMA_CPP_BASE_URL", "")
GENERATE_TIMEOUT = 300.0


class LLMError(Exception):
    pass


def _ollama_generate(prompt: str, model: str, options: dict[str, Any] | None = None) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options
    with httpx.Client(timeout=GENERATE_TIMEOUT) as client:
        try:
            r = client.post(url, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise LLMError("Ollama not reachable. Is it running? (e.g. ollama serve)") from e
        except httpx.TimeoutException as e:
            raise LLMError("Ollama request timed out.") from e
        if r.status_code == 404:
            raise LLMError(f"Model {model!r} not found. Pull it with: ollama pull {model}")
        r.raise_for_status()
        data = r.json()
    response = data.get("response")
    if response is None:
        raise LLMError("Ollama returned no response text.")
    return str(response).strip()


def _llama_cpp_generate(prompt: str, options: dict[str, Any] | None = None) -> str:
    url = f"{LLAMA_CPP_BASE_URL.rstrip('/')}/completion"
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": 2048,
        "stream": False,
    }
    if options:
        if "num_predict" in options:
            payload["n_predict"] = options["num_predict"]
        if "temperature" in options:
            payload["temperature"] = options["temperature"]
        if "top_p" in options:
            payload["top_p"] = options["top_p"]
    with httpx.Client(timeout=GENERATE_TIMEOUT) as client:
        try:
            r = client.post(url, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise LLMError(
                "llama.cpp server not reachable. Is it running at LLAMA_CPP_BASE_URL?"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError("llama.cpp request timed out.") from e
        r.raise_for_status()
        data = r.json()
    content = data.get("content")
    if content is None:
        raise LLMError("llama.cpp returned no content.")
    return str(content).strip()


def generate_story(prompt: str, model_override: str | None = None) -> str:
    """
    Generate story text using Ollama or llama.cpp (if LLAMA_CPP_BASE_URL is set).
    Returns the generated text; raises LLMError on failure.
    """
    if LLAMA_CPP_BASE_URL:
        return _llama_cpp_generate(prompt)
    model = model_override or OLLAMA_MODEL
    return _ollama_generate(prompt, model)
