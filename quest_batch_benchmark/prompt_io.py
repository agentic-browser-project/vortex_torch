"""Load the benchmark request and render it into a chat-templated prompt string."""
from __future__ import annotations

import json
from typing import List


def load_request_messages(request_path: str) -> List[dict]:
    """Return the chat `messages` list from a request.json file."""
    with open(request_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["messages"]


def build_prompt_text(request_path: str, tokenizer) -> str:
    """Render the request through the model chat template into a prompt string.

    `tokenize=False` returns the rendered template text -- the form `sgl.Engine`
    ingests; `add_generation_prompt=True` appends the assistant turn marker so
    the model is positioned to decode the first output token. Mirrors
    `prepare_prompts` in the sgl baseline `measure_batch_latency_offline.py`.
    """
    messages = load_request_messages(request_path)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
