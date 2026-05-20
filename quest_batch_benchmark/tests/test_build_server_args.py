import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark_quest_tpot import build_parser, build_server_args


def _args(attention):
    return build_parser().parse_args(["--attention", attention])


def test_build_server_args_dense_mode():
    sa = build_server_args(_args("dense"), n_input_tokens=9661)
    assert sa.model_path == _args("dense").model_path
    assert sa.disable_radix_cache is True
    assert sa.enable_vortex_sparsity is False
    assert sa.tp_size == 1


def test_build_server_args_quest_mode_enables_sparsity():
    sa = build_server_args(_args("quest"), n_input_tokens=9661)
    assert sa.enable_vortex_sparsity is True
    assert sa.vortex_module_name == "gqa_quest_sparse_attention"
    assert sa.disable_radix_cache is True
