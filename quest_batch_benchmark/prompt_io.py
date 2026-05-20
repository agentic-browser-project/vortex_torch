"""Load the benchmark request and turn it into model input token ids."""
from __future__ import annotations

import json
from typing import List


def load_request_messages(request_path: str) -> List[dict]:
    """Return the chat `messages` list from a request.json file."""
    with open(request_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["messages"]


def build_input_ids(request_path: str, tokenizer) -> List[int]:
    """Apply the model chat template to the request and return token ids.

    `add_generation_prompt=True` appends the assistant turn marker so the
    model is positioned to decode the first output token.
    """
    messages = load_request_messages(request_path)
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )
    # Some tokenizer versions return a BatchEncoding (UserDict); unwrap if needed.
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return [int(t) for t in ids]
