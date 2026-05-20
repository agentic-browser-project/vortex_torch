import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_io import build_input_ids, load_request_messages

MODEL_DIR = "/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B"
REQUEST = os.path.join(os.path.dirname(__file__), "..", "request.json")


def test_load_request_messages_has_system_and_user():
    messages = load_request_messages(REQUEST)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all(isinstance(m["content"], str) and m["content"] for m in messages)


def test_build_input_ids_returns_long_token_list():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = build_input_ids(REQUEST, tok)
    assert isinstance(ids, list)
    assert all(isinstance(t, int) for t in ids)
    # request_005 is ~28.6K chars of system+user text -> several thousand tokens
    assert 3000 < len(ids) < 20000
