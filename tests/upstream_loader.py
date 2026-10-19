"""Load pinned upstream SUT / MoM source files for parity tests.

SUT's layer files import fairseq only for small utilities. Rather than installing fairseq (which no
longer installs cleanly on recent Python/PyTorch), we register minimal stand-ins whose behaviour
equals what those files use: FairseqDropout (= nn.Dropout), quant_noise (identity when p=0),
with_incremental_state (adds get/set_incremental_state), utils.softmax (float32 softmax),
utils.get_activation_fn('gelu') (= exact-erf GELU), utils.safe_getattr, LayerNorm (= nn.LayerNorm).
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from moebase import paths


def _stub_fairseq():
    if "fairseq" in sys.modules and getattr(sys.modules["fairseq"], "_moebase_stub", False):
        return
    fs = types.ModuleType("fairseq"); fs._moebase_stub = True
    utils = types.ModuleType("fairseq.utils")
    utils.softmax = lambda x, dim, onnx_trace=False: F.softmax(x.float(), dim=dim, dtype=torch.float32)
    utils.get_activation_fn = lambda activation: {"gelu": lambda x: F.gelu(x.float()).type_as(x), "relu": F.relu}[activation]
    utils.safe_getattr = lambda obj, k, default=None: getattr(obj, k, default)
    fs.utils = utils

    inc = types.ModuleType("fairseq.incremental_decoding_utils")

    def with_incremental_state(cls):
        def get_incremental_state(self, incremental_state, key):
            return None if incremental_state is None else incremental_state.get(key)

        def set_incremental_state(self, incremental_state, key, value):
            if incremental_state is not None:
                incremental_state[key] = value
            return incremental_state
        cls.get_incremental_state = get_incremental_state
        cls.set_incremental_state = set_incremental_state
        return cls
    inc.with_incremental_state = with_incremental_state

    modules = types.ModuleType("fairseq.modules")

    class LayerNorm(nn.LayerNorm):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
            super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    modules.LayerNorm = LayerNorm
    fd = types.ModuleType("fairseq.modules.fairseq_dropout")

    class FairseqDropout(nn.Dropout):
        def __init__(self, p, module_name=None):
            super().__init__(p)
    fd.FairseqDropout = FairseqDropout
    qn = types.ModuleType("fairseq.modules.quant_noise")
    qn.quant_noise = lambda module, p, block_size: module
    models = types.ModuleType("fairseq.models")
    mt = types.ModuleType("fairseq.models.transformer")

    class TransformerConfig:  # only from_namespace is referenced by the legacy wrappers
        @staticmethod
        def from_namespace(ns):
            return ns
    mt.TransformerConfig = TransformerConfig
    for name, mod in {"fairseq": fs, "fairseq.utils": utils, "fairseq.incremental_decoding_utils": inc,
                      "fairseq.modules": modules, "fairseq.modules.fairseq_dropout": fd,
                      "fairseq.modules.quant_noise": qn, "fairseq.models": models,
                      "fairseq.models.transformer": mt}.items():
        sys.modules[name] = mod


def _load_package(pkg_name: str, pkg_dir: Path):
    spec = importlib.util.spec_from_file_location(pkg_name, pkg_dir / "__init__.py",
                                                  submodule_search_locations=[str(pkg_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_file(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def sut_available() -> bool:
    return (paths.upstream_dir("SUT") / "sut_layer" / "moa_attention.py").exists()


def load_sut():
    """Returns namespace with upstream MoE, MultiheadAttention, encoder/decoder layers, ACTWrapper."""
    _stub_fairseq()
    root = paths.upstream_dir("SUT")
    pkg = _load_package("upstream_sut_layer", root / "sut_layer")
    pe = importlib.import_module("upstream_sut_layer.parallel_linear.parallel_experts")
    att = importlib.import_module("upstream_sut_layer.moa_attention")
    layer = importlib.import_module("upstream_sut_layer.sut_transformer_layer")
    halting = _load_file("upstream_sut_halting", root / "halting.py")
    return types.SimpleNamespace(MoE=pe.MoE, MultiheadAttention=att.MultiheadAttention,
                                 EncoderLayer=layer.TransformerEncoderLayerBase,
                                 DecoderLayer=layer.TransformerDecoderLayerBase, ACTWrapper=halting.ACTWrapper)


def sut_cfg_namespace(b):
    """fairseq-style cfg for the upstream layer classes, from our SUTBlockConfig."""
    ns = types.SimpleNamespace
    enc = ns(embed_dim=b.embed_dim, normalize_before=b.normalize_before, attn_num_expert=b.attn_num_expert,
             attn_k=b.attn_k, attn_expert_dim=b.attn_expert_dim, ff_num_expert=b.ff_num_expert, ff_k=b.ff_k,
             ff_expert_dim=b.ff_expert_dim)
    dec = ns(**vars(enc))
    return ns(encoder=enc, decoder=dec, quant_noise=ns(pq=0.0, pq_block_size=8), export=False,
              dropout=b.dropout, activation_fn=b.activation_fn, activation_dropout=b.activation_dropout,
              relu_dropout=0.0, attention_dropout=b.attention_dropout, head_dim=b.head_dim,
              cvloss=b.cvloss, switchloss=b.switchloss, zloss=b.zloss, miloss=b.miloss,
              sample_topk=b.sample_topk, gating_dropout=b.gating_dropout, cross_self_attention=False)


def mom_available() -> bool:
    return (paths.upstream_dir("Mixture-of-Modules") / "mom" / "model.py").exists()


def _stub_transformers():
    """Minimal stand-in for the parts of `transformers` that MoM's files import.

    Only Conv1D, the GELU variants and PretrainedConfig's attribute_map affect numerics; they are copied
    from transformers (pytorch_utils.Conv1D, activations.NewGELUActivation, configuration_utils). Everything
    else (docstring decorators, output dataclasses, ONNX config, tokenizer types) is inert at import time.
    """
    import dataclasses
    from collections import OrderedDict

    def mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    tf = mod("transformers"); tf._moebase_stub = True
    utils = mod("transformers.utils")
    logging = mod("transformers.utils.logging")

    class _Logger:
        def __getattr__(self, k):
            return lambda *a, **kw: None
    logging.get_logger = lambda *a, **k: _Logger()
    utils.logging = logging

    class ModelOutput(OrderedDict):
        def __post_init__(self):
            for f in dataclasses.fields(self):
                v = getattr(self, f.name)
                if v is not None:
                    self[f.name] = v

        def __getitem__(self, k):
            if isinstance(k, str):
                return dict(self.items())[k]
            return self.to_tuple()[k]

        def to_tuple(self):
            return tuple(self[k] for k in self.keys())
    utils.ModelOutput = ModelOutput
    ident = lambda *a, **k: (lambda f: f)  # noqa: E731
    for n in ("add_code_sample_docstrings", "add_start_docstrings", "add_start_docstrings_to_model_forward",
              "replace_return_docstrings"):
        setattr(utils, n, ident)
    utils.is_flash_attn_2_available = lambda: False
    utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
    mpu = mod("transformers.utils.model_parallel_utils")
    mpu.assert_device_map = lambda *a, **k: None
    mpu.get_device_map = lambda *a, **k: None

    act = mod("transformers.activations")

    class NewGELU(nn.Module):  # transformers.activations.NewGELUActivation
        def forward(self, x):
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    act.ACT2FN = {"gelu_new": NewGELU(), "gelu": nn.GELU(), "relu": nn.ReLU()}

    pu = mod("transformers.pytorch_utils")

    class Conv1D(nn.Module):  # transformers.pytorch_utils.Conv1D
        def __init__(self, nf, nx):
            super().__init__()
            self.nf = nf
            self.weight = nn.Parameter(torch.empty(nx, nf))
            self.bias = nn.Parameter(torch.zeros(nf))
            nn.init.normal_(self.weight, std=0.02)

        def forward(self, x):
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            return x.view(size_out)
    pu.Conv1D = Conv1D
    pu.find_pruneable_heads_and_indices = pu.prune_conv1d_layer = lambda *a, **k: None

    mo = mod("transformers.modeling_outputs")
    for n in ("BaseModelOutputWithPastAndCrossAttentions", "CausalLMOutputWithCrossAttentions",
              "QuestionAnsweringModelOutput", "SequenceClassifierOutputWithPast", "TokenClassifierOutput"):
        setattr(mo, n, type(n, (ModelOutput,), {}))
    mu = mod("transformers.modeling_utils")

    class PreTrainedModel(nn.Module):
        def __init__(self, config, *a, **k):
            super().__init__()
            self.config = config

        def post_init(self):
            pass
    mu.PreTrainedModel = PreTrainedModel
    mu.SequenceSummary = nn.Module

    cu = mod("transformers.configuration_utils")

    class PretrainedConfig:
        attribute_map = {}

        def __setattr__(self, k, v):
            k = type(self).attribute_map.get(k, k)
            object.__setattr__(self, k, v)

        def __getattribute__(self, k):
            if k != "attribute_map" and k in type(self).attribute_map:
                k = type(self).attribute_map[k]
            return object.__getattribute__(self, k)

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    cu.PretrainedConfig = PretrainedConfig
    onnx = mod("transformers.onnx")
    onnx.OnnxConfigWithPast = type("OnnxConfigWithPast", (), {})
    onnx.PatchingSpec = object
    tf.PreTrainedTokenizer = object
    tf.TensorType = object
    tf.is_torch_available = lambda: True
    tf.utils, tf.activations, tf.pytorch_utils = utils, act, pu


def load_mom():
    """Import upstream mom/model.py. Uses `transformers` when installed, otherwise a minimal stand-in
    (_stub_transformers). The missing configuration_gpt_mot module (finding M2) is aliased to the
    shipped mom/config.py."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        _stub_transformers()
    root = paths.upstream_dir("Mixture-of-Modules") / "mom"
    spec = importlib.util.spec_from_loader("upstream_mom", loader=None, is_package=True)
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [str(root)]
    sys.modules["upstream_mom"] = pkg
    cfg = _load_file("upstream_mom.config", root / "config.py")
    sys.modules["upstream_mom.configuration_gpt_mot"] = cfg
    _load_file("upstream_mom.utils", root / "utils.py")
    model = _load_file("upstream_mom.model", root / "model.py")
    return types.SimpleNamespace(GPT2Config=cfg.GPT2Config, model=model, utils=sys.modules["upstream_mom.utils"])
