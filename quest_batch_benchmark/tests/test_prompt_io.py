import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_io import build_prompt_text, load_request_messages

MODEL_DIR = "/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B"
REQUEST = os.path.join(os.path.dirname(__file__), "..", "request.json")


def test_load_request_messages_has_system_and_user():
    messages = load_request_messages(REQUEST)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all(isinstance(m["content"], str) and m["content"] for m in messages)


def test_build_prompt_text_applies_chat_template():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    text = build_prompt_text(REQUEST, tok)
    assert isinstance(text, str)
    # the Qwen3 chat template wraps each role in <|im_start|>/<|im_end|>
    assert "<|im_start|>system" in text
    assert "<|im_start|>user" in text
    # add_generation_prompt=True appends the assistant turn marker
    assert "<|im_start|>assistant" in text
    # request_005 is ~28.6K chars of system+user text
    assert len(text) > 10000
