"""Mixture-of-Modules (MoM) model.

Port of Mixture-of-Modules@2ed1d28 ``mom/model.py`` (MoTAttention, MoTMLP, MoTBlock, MoTLayer,
GPT2MoTModel, GPT2MoTLMHeadModel) without the HuggingFace PreTrainedModel machinery. Parameter
names are kept identical (``transformer.h.{i}.mot_attn.c_attn.{m}.weight`` ...) so that the
upstream conversion map (convert.py) and upstream state dicts apply unchanged.

Model layout (upstream ``GPT2MoTModel.__init__``):
    [num_bottom_layers x MoTLayer] + chunks x [MoTBlock (+ MoTLayer anchor if require_anchor_layer)]
    + [num_top_layers x MoTLayer]
MoTLayer = an ordinary pre-LN GPT-2 block. MoTBlock = one pool of ``num_attn`` attention modules
and ``num_ffn`` FFN modules applied for ``depth_per_chunk`` assembly steps, routed per token.

Assembly in one step of MoTBlock (``routing_top_k`` = K, SKIP = last router class):
    qkv(x)  = sum_{m in TopK, m != SKIP} c_attn_m(LN_m(x))      (scores used only if weighted_qkv)
    a       = causal multi-head attention(q, k, v)               (one attention computation)
    o       = sum_{m in TopK, m != SKIP} r_m * c_proj_m(a)
    u       = x + o
    f       = sum_{m in TopK, m != SKIP} r_m * FFN_m(LN_m(u))
    x'      = u + f
Each token picks its own modules; a token's key/value come from *its own* selected projections.
A token that selects only SKIP gets q = k = v = 0 (it still appears as a zero key/value to later
tokens) and an exactly zero update.

Removed: KV cache (upstream reuses one ``layer_past`` across all depths, which is incorrect for
generation and unused in training), flash-attention, model parallel, gradient checkpointing,
head masks, cross-attention, the ``routing_all_possible_path`` *module-skipping* behaviour is kept.
Added: ``causal`` flag (default True). ``causal=False`` is the labelled encoder adaptation used by the
controlled common interface; it is not part of upstream MoM.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .routers import STE, get_router, load_balancing_loss_func


@dataclass
class MoMConfig:
    # GPT-2 part
    vocab_size: int = 32
    n_positions: int = 128
    n_embd: int = 128
    n_layer: int = 6  # donor depth; used only for the GPT-2 c_proj init scale
    n_head: int = 4
    n_inner: Optional[int] = 512
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    # MoM part (upstream names)
    chunks: int = 1
    num_bottom_layers: int = 1
    num_top_layers: int = 0
    num_attn: int = 4
    num_ffn: int = 4
    depth_per_chunk: Optional[int] = 4
    depth_multiplier: float = 4
    require_anchor_layer: bool = True
    router_type: str = "gru"
    router_dim: int = 256
    router_softmax_temperature: float = 1.0
    enable_pause: bool = True
    routing_top_k: int = 1
    weighted_qkv: bool = False
    routing_all_possible_path: bool = False
    use_ste: bool = False
    output_router_logits: bool = True
    router_aux_loss_coef: float = 0.01
    depthwise_bal_loss: bool = False
    bind_ffn_attn: bool = False
    # adaptation flags (not upstream)
    causal: bool = True
    num_classes: int = 7

    @property
    def hidden_size(self):
        return self.n_embd

    def to_dict(self):
        return asdict(self)

    def donor_layers(self) -> int:
        """Number of vanilla layers consumed by get_layer_map()."""
        per_chunk = max(self.num_ffn, self.num_attn) + (1 if self.require_anchor_layer else 0)
        return self.num_bottom_layers + self.chunks * per_chunk + self.num_top_layers

    def block_applications(self) -> int:
        depth = self.depth_per_chunk if self.depth_per_chunk is not None else int(self.num_ffn * self.depth_multiplier)
        return (self.num_bottom_layers + self.chunks * (depth + (1 if self.require_anchor_layer else 0))
                + self.num_top_layers)


ACT = {
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "gelu": F.gelu,
    "relu": F.relu,
}


class Conv1D(nn.Module):
    """transformers.pytorch_utils.Conv1D: y = x @ W + b with W of shape (nx, nf)."""

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


class MoTAttention(nn.Module):
    def __init__(self, config: MoMConfig, chunk_idx: int, is_anchor_layer: bool = False):
        super().__init__()
        self.config = config
        self.router_type = config.router_type
        self.chunk_idx = chunk_idx
        self.num_attn = config.num_attn
        self.is_anchor_layer = is_anchor_layer or self.num_attn == 1
        max_positions = config.n_positions
        self.register_buffer("bias", torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))
                             .view(1, 1, max_positions, max_positions), persistent=False)
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        assert self.head_dim * self.num_heads == self.embed_dim
        self.scale_attn_weights = config.scale_attn_weights
        if self.is_anchor_layer:
            self.attention_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.attention_layernorm = nn.ModuleList(
                [nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon) for _ in range(self.num_attn)])
            self.c_attn = nn.ModuleList([Conv1D(3 * self.embed_dim, self.embed_dim) for _ in range(self.num_attn)])
            self.c_proj = nn.ModuleList([Conv1D(self.embed_dim, self.embed_dim) for _ in range(self.num_attn)])
            self.router = get_router(config)
        self.ste = STE.apply
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full([], value.size(-1) ** 0.5, dtype=attn_weights.dtype,
                                                     device=attn_weights.device)
        if self.config.causal:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
            mask_value = torch.full([], torch.finfo(attn_weights.dtype).min, dtype=attn_weights.dtype,
                                    device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        return torch.matmul(attn_weights, value), attn_weights

    @staticmethod
    def _split_heads(tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        return tensor.view(new_shape).permute(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))

    def _route(self, hidden_states, routing_states):
        bl = hidden_states.size(0)
        if "naive" in self.router_type:
            routing_score = self.router(hidden_states).view(bl, -1)
        elif "gru" in self.router_type or "dev" in self.router_type:
            routing_score, routing_states = self.router(hidden_states, routing_states)
            routing_score = routing_score.view(bl, -1)
        else:
            raise NotImplementedError(self.router_type)
        num_classes = routing_score.shape[-1]
        top_score, top_ids = torch.topk(routing_score, min(self.config.routing_top_k, num_classes), dim=-1)
        if "softmax" in self.router_type:
            top_score = top_score / top_score.sum(dim=-1, keepdim=True)
        expert_mask = F.one_hot(top_ids, num_classes=num_classes).permute(2, 1, 0)
        return routing_score, routing_states, top_score, top_ids, expert_mask

    def forward(self, hidden_states, attention_mask=None, routing_states=None, output_attentions=False):
        routing_score = None
        bsz, q_len, _ = hidden_states.size()
        hidden_states = hidden_states.view(bsz * q_len, -1)
        route = None
        if self.is_anchor_layer:
            qkv_states = self.c_attn(self.attention_layernorm(hidden_states))
        else:
            routing_score, routing_states, top_score, top_ids, expert_mask = self._route(hidden_states, routing_states)
            route = top_ids
            qkv_states = torch.zeros((bsz * q_len, self.embed_dim * 3), dtype=hidden_states.dtype,
                                     device=hidden_states.device)
            for module_idx in range(self.num_attn):
                if self.config.routing_all_possible_path:
                    current_state = hidden_states
                    selected_score = routing_score[:, module_idx].unsqueeze(1)
                    idx = None
                else:
                    idx2, idx = torch.where(expert_mask[module_idx])
                    if len(idx) == 0:
                        continue
                    current_state = hidden_states[idx]
                    selected_score = top_score[idx, idx2, None]
                current_state = self.attention_layernorm[module_idx](current_state)
                current_state = self.c_attn[module_idx](current_state)
                if self.config.weighted_qkv:
                    current_state = current_state * (self.ste(selected_score) if self.config.use_ste else selected_score)
                if idx is None:
                    qkv_states = qkv_states + current_state
                else:
                    qkv_states = qkv_states.index_add(0, idx, current_state)  # upstream: qkv_states[idx] += ...

        query, key, value = qkv_states.view(bsz, q_len, -1).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim).view(bsz * q_len, -1)

        if self.is_anchor_layer:
            output_states = self.c_proj(attn_output).reshape(bsz, q_len, -1)
        else:
            output_states = torch.zeros((bsz * q_len, self.embed_dim), dtype=attn_output.dtype, device=attn_output.device)
            for module_idx in range(self.num_attn):
                if self.config.routing_all_possible_path:
                    current_state = attn_output
                    selected_score = routing_score[:, module_idx].unsqueeze(1)
                    idx = None
                else:
                    idx2, idx = torch.where(expert_mask[module_idx])
                    if len(idx) == 0:
                        continue
                    current_state = attn_output[idx]
                    selected_score = top_score[idx, idx2, None]
                o_state = self.c_proj[module_idx](current_state)
                o_state = o_state * (self.ste(selected_score) if self.config.use_ste else selected_score)
                if idx is None:
                    output_states = output_states + o_state
                else:
                    output_states = output_states.index_add(0, idx, o_state)
            output_states = output_states.reshape(bsz, q_len, -1)
            routing_score = routing_score.reshape(bsz, q_len, -1)
        output_states = self.resid_dropout(output_states)
        return output_states, (attn_weights if output_attentions else None), routing_score, routing_states, route


class MoTMLP(nn.Module):
    def __init__(self, config: MoMConfig, chunk_idx: int, is_anchor_layer: bool = False):
        super().__init__()
        self.config = config
        self.router_type = config.router_type
        self.hidden_size = config.n_embd
        self.intermediate_size = config.n_inner if config.n_inner is not None else 4 * self.hidden_size
        self.num_ffn = config.num_ffn
        self.is_anchor_layer = is_anchor_layer or self.num_ffn == 1
        if self.is_anchor_layer:
            self.ffn_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
            self.c_fc = Conv1D(self.intermediate_size, self.hidden_size)
            self.c_proj = Conv1D(self.hidden_size, self.intermediate_size)
        else:
            self.ffn_layernorm = nn.ModuleList(
                [nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon) for _ in range(self.num_ffn)])
            self.c_fc = nn.ModuleList([Conv1D(self.intermediate_size, self.hidden_size) for _ in range(self.num_ffn)])
            self.c_proj = nn.ModuleList([Conv1D(self.hidden_size, self.intermediate_size) for _ in range(self.num_ffn)])
            self.router = get_router(config)
        self.ste = STE.apply
        self.act = ACT[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x, routing_states=None):
        bsz, q_len, _ = x.shape
        x = x.view(bsz * q_len, -1)
        route = None
        if self.is_anchor_layer:
            out = self.c_proj(self.act(self.c_fc(self.ffn_layernorm(x))))
            out = self.dropout(out).reshape(bsz, q_len, -1)
            return out, None, routing_states, None
        if "naive" in self.router_type:
            routing_score = self.router(x).view(bsz * q_len, -1)
        else:
            routing_score, routing_states = self.router(x, routing_states)
            routing_score = routing_score.view(bsz * q_len, -1)
        num_classes = routing_score.shape[-1]
        top_score, top_ids = torch.topk(routing_score, min(self.config.routing_top_k, num_classes), dim=-1)
        if "softmax" in self.router_type:
            top_score = top_score / top_score.sum(dim=-1, keepdim=True)
        expert_mask = F.one_hot(top_ids, num_classes=num_classes).permute(2, 1, 0)
        route = top_ids
        output_states = torch.zeros((bsz * q_len, self.hidden_size), dtype=x.dtype, device=x.device)
        for module_idx in range(self.num_ffn):
            if self.config.routing_all_possible_path:
                current_state = x
                selected_score = routing_score[:, module_idx].unsqueeze(1)
                idx = None
            else:
                idx2, idx = torch.where(expert_mask[module_idx])
                if len(idx) == 0:
                    continue
                current_state = x[idx]
                selected_score = top_score[idx, idx2, None]
            current_state = self.ffn_layernorm[module_idx](current_state)
            current_state = self.c_proj[module_idx](self.act(self.c_fc[module_idx](current_state)))
            current_state = current_state * (self.ste(selected_score) if self.config.use_ste else selected_score)
            output_states = output_states + current_state if idx is None else output_states.index_add(0, idx, current_state)
        # NB: upstream applies no dropout on the routed FFN path (only on the anchor path).
        return output_states.reshape(bsz, q_len, -1), routing_score.reshape(bsz, q_len, -1), routing_states, route


class MoTBlock(nn.Module):
    def __init__(self, config: MoMConfig, chunk_idx: int):
        super().__init__()
        self.chunk_idx = chunk_idx
        self.max_depth = config.depth_per_chunk if config.depth_per_chunk is not None else int(
            config.num_ffn * config.depth_multiplier)
        self.mot_attn = MoTAttention(config, chunk_idx)
        self.mot_ffn = MoTMLP(config, chunk_idx)

    def forward(self, hidden_states, attention_mask=None, output_router_logits=False):
        block_router_logits = () if output_router_logits else None
        routes = []
        routing_states_attn = None  # reset at every block: GRU state lives inside one chunk
        routing_states_ffn = None
        for _ in range(self.max_depth):
            residual = hidden_states
            hidden_states, _, attn_routing_score, routing_states_attn, r_a = self.mot_attn(
                hidden_states, attention_mask=attention_mask, routing_states=routing_states_attn)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states, ffn_routing_score, routing_states_ffn, r_f = self.mot_ffn(hidden_states, routing_states_ffn)
            hidden_states = residual + hidden_states
            if output_router_logits:
                block_router_logits += ((attn_routing_score, ffn_routing_score),)
            routes.append((r_a, r_f))
        return hidden_states, block_router_logits, routes


class MoTLayer(nn.Module):
    def __init__(self, config: MoMConfig, chunk_idx: int):
        super().__init__()
        self.chunk_idx = chunk_idx
        self.mot_attn = MoTAttention(config, chunk_idx, is_anchor_layer=True)
        self.mot_ffn = MoTMLP(config, chunk_idx, is_anchor_layer=True)

    def forward(self, hidden_states, attention_mask=None, output_router_logits=False):
        residual = hidden_states
        hidden_states, _, _, _, _ = self.mot_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states, _, _, _ = self.mot_ffn(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, None, []


def gpt2_init_weights(model: nn.Module, initializer_range: float, n_layer: int):
    """GPT2PreTrainedModel._init_weights applied to every submodule (HF post_init).

    nn.Module.apply visits children before parents, so the c_proj rescale done at the parent is not
    overwritten by the child's generic init. Reversed pre-order preserves that property.
    """
    for module in reversed(list(model.modules())):
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # only a *direct* parameter named exactly 'c_proj.weight' is rescaled (anchor / dense layers,
        # not the per-module ModuleList entries 'c_proj.0.weight' ...), as in upstream
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                p.data.normal_(mean=0.0, std=(initializer_range / math.sqrt(2 * n_layer)))


class GPT2MoTModel(nn.Module):
    def __init__(self, config: MoMConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        if config.bind_ffn_attn:
            assert config.num_attn == config.num_ffn
        self.h = nn.ModuleList()
        for i in range(config.num_bottom_layers):
            self.h.append(MoTLayer(config, i))
        for i in range(config.chunks):
            self.h.append(MoTBlock(config, chunk_idx=i))
            if config.require_anchor_layer:
                self.h.append(MoTLayer(config, i + config.num_bottom_layers))
        for i in range(config.num_top_layers):
            self.h.append(MoTLayer(config, i + config.num_bottom_layers + config.chunks))
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask=None, output_router_logits=None):
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits
        bsz, seq_len = input_ids.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        if attention_mask is not None:
            am = attention_mask.view(bsz, -1)[:, None, None, :].to(dtype=self.wte.weight.dtype)
            am = (1.0 - am) * torch.finfo(self.wte.weight.dtype).min
        else:
            am = None
        hidden_states = self.drop(self.wte(input_ids) + self.wpe(position_ids))
        all_router_logits = () if output_router_logits else None
        all_routes = []
        for block in self.h:
            hidden_states, router_logits, routes = block(hidden_states, attention_mask=am,
                                                         output_router_logits=output_router_logits)
            if output_router_logits and router_logits is not None:
                all_router_logits += router_logits
            all_routes.extend(routes)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, all_router_logits, all_routes


def mom_aux_loss(config: MoMConfig, routing_scores, attention_mask, dtype, device):
    """Upstream GPT2MoTLMHeadModel.forward aux-loss block."""
    if not routing_scores or routing_scores[0][0] is None:
        return None
    num_modules_attn = routing_scores[0][0].shape[-1] if routing_scores[0][0] is not None else 1
    num_modules_ffn = routing_scores[0][1].shape[-1] if routing_scores[0][1] is not None else 1
    aux_loss = torch.tensor(0, dtype=dtype, device=device)
    if config.depthwise_bal_loss:
        for r in routing_scores:
            if num_modules_attn > 1:
                aux_loss = aux_loss + load_balancing_loss_func(r[0].view(-1, num_modules_attn), num_modules_attn, attention_mask)
            if num_modules_ffn > 1:
                aux_loss = aux_loss + load_balancing_loss_func(r[1].view(-1, num_modules_ffn), num_modules_ffn, attention_mask)
        aux_loss = aux_loss / len(routing_scores)
    else:
        if num_modules_attn > 1:
            aux_loss = aux_loss + load_balancing_loss_func(
                tuple(r[0].view(-1, num_modules_attn) for r in routing_scores), num_modules_attn, attention_mask)
        if num_modules_ffn > 1:
            aux_loss = aux_loss + load_balancing_loss_func(
                tuple(r[1].view(-1, num_modules_ffn) for r in routing_scores), num_modules_ffn, attention_mask)
    return aux_loss


class MoMCausalLM(nn.Module):
    """GPT2MoTLMHeadModel: tied LM head, shifted CE with ignore_index -100, separate aux loss."""

    def __init__(self, config: MoMConfig):
        super().__init__()
        self.config = config
        self.transformer = GPT2MoTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        gpt2_init_weights(self, config.initializer_range, config.n_layer)
        self.lm_head.weight = self.transformer.wte.weight  # tie_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden, routing_scores, routes = self.transformer(input_ids, attention_mask=attention_mask)
        lm_logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        aux_loss = None
        if labels is not None and self.config.output_router_logits:
            aux_loss = mom_aux_loss(self.config, routing_scores, attention_mask, loss.dtype, loss.device)
        return {"loss": loss, "aux_loss": aux_loss, "logits": lm_logits, "routing_scores": routing_scores,
                "routes": routes}


class MoMClassifier(nn.Module):
    """Controlled-interface adaptation (labelled): bidirectional MoM encoder + readout at <cls>."""

    def __init__(self, config: MoMConfig):
        super().__init__()
        assert not config.causal, "use causal=False for the encoder adaptation"
        self.config = config
        self.transformer = GPT2MoTModel(config)
        self.head = nn.Linear(config.n_embd, config.num_classes)
        gpt2_init_weights(self, config.initializer_range, config.n_layer)

    def forward(self, input_ids, attention_mask=None):
        hidden, routing_scores, routes = self.transformer(input_ids, attention_mask=attention_mask)
        logits = self.head(hidden[:, 0])
        aux = None
        if self.config.output_router_logits:
            aux = mom_aux_loss(self.config, routing_scores, attention_mask, logits.dtype, logits.device)
        return {"logits": logits, "aux_loss": aux, "routing_scores": routing_scores, "routes": routes}
