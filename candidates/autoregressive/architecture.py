from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module, control_patterns: tuple[str, ...]) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            is_control = any(pattern in name for pattern in control_patterns)
            if (param.ndim < 2 or is_control) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        # Validation runs under inference_mode and may populate the cache before
        # training starts. Clone on return so cached inference tensors are never
        # reused directly in autograd-tracked CPU subset training.
        return self._cos_cached.to(dtype=dtype).clone(), self._sin_cached.to(dtype=dtype).clone()


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        attn_norm_mode: str,
        attn_norm_eps: float,
        sparse_attn_mode: str,
        sparse_attn_window: int,
        sparse_attn_global_tokens: int,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        if attn_norm_mode not in {"baseline", "qk_rmsnorm", "qk_norm_per_head"}:
            raise ValueError(f"Unsupported ATTN_NORM_MODE={attn_norm_mode!r}")
        if attn_norm_eps <= 0.0:
            raise ValueError(f"ATTN_NORM_EPS must be > 0, got {attn_norm_eps}")
        if sparse_attn_mode not in {"dense", "local_global"}:
            raise ValueError(f"Unsupported SPARSE_ATTN_MODE={sparse_attn_mode!r}")
        if sparse_attn_window < 1:
            raise ValueError(f"SPARSE_ATTN_WINDOW must be >= 1, got {sparse_attn_window}")
        if sparse_attn_global_tokens < 0:
            raise ValueError(
                f"SPARSE_ATTN_GLOBAL_TOKENS must be >= 0, got {sparse_attn_global_tokens}"
            )
        self.attn_norm_mode = attn_norm_mode
        self.attn_norm_eps = attn_norm_eps
        self.sparse_attn_mode = sparse_attn_mode
        self.sparse_attn_window = sparse_attn_window
        self.sparse_attn_global_tokens = sparse_attn_global_tokens
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self._mask_seq_len_cached = 0
        self._mask_cached: Tensor | None = None

    def _normalize_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        if self.attn_norm_mode == "qk_rmsnorm":
            q = F.rms_norm(q, (q.size(-1),), eps=self.attn_norm_eps)
            k = F.rms_norm(k, (k.size(-1),), eps=self.attn_norm_eps)
            return q, k
        if self.attn_norm_mode == "qk_norm_per_head":
            q = F.normalize(q, p=2.0, dim=-1, eps=self.attn_norm_eps)
            k = F.normalize(k, p=2.0, dim=-1, eps=self.attn_norm_eps)
            return q, k
        return q, k

    def _local_global_mask(self, seqlen: int, device: torch.device) -> Tensor:
        if (
            self._mask_cached is not None
            and self._mask_seq_len_cached == seqlen
            and self._mask_cached.device == device
        ):
            return self._mask_cached
        positions = torch.arange(seqlen, device=device)
        query_pos = positions[:, None]
        key_pos = positions[None, :]
        causal = key_pos <= query_pos
        local = key_pos >= (query_pos - self.sparse_attn_window + 1)
        global_key = key_pos < self.sparse_attn_global_tokens
        mask = (causal & (local | global_key))[None, None, :, :]
        self._mask_cached = mask
        self._mask_seq_len_cached = seqlen
        return mask

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = self._normalize_qk(q, k)
        if self.attn_norm_mode == "baseline":
            q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.sparse_attn_mode == "dense":
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=self._local_global_mask(seqlen, x.device),
                is_causal=False,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(
        self,
        dim: int,
        mlp_mult: int,
        activation_mode: str,
        swiglu_clamp_enabled: bool,
        swiglu_linear_clamp_min: float,
        swiglu_linear_clamp_max: float,
        swiglu_gate_clamp_max: float,
    ):
        super().__init__()
        hidden = mlp_mult * dim
        if activation_mode not in {"relu2", "swiglu"}:
            raise ValueError(f"Unsupported ACTIVATION_MODE: {activation_mode}")
        if swiglu_linear_clamp_min > swiglu_linear_clamp_max:
            raise ValueError(
                "SWIGLU_LINEAR_CLAMP_MIN must be <= SWIGLU_LINEAR_CLAMP_MAX"
            )
        if swiglu_gate_clamp_max <= 0.0:
            raise ValueError("SWIGLU_GATE_CLAMP_MAX must be positive")
        self.activation_mode = activation_mode
        self.swiglu_clamp_enabled = swiglu_clamp_enabled
        self.swiglu_linear_clamp_min = swiglu_linear_clamp_min
        self.swiglu_linear_clamp_max = swiglu_linear_clamp_max
        self.swiglu_gate_clamp_max = swiglu_gate_clamp_max
        self.fc = CastedLinear(dim, hidden * (2 if activation_mode == "swiglu" else 1), bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_mode == "swiglu":
            linear, gate = self.fc(x).chunk(2, dim=-1)
            if self.swiglu_clamp_enabled:
                linear = torch.clamp(
                    linear,
                    min=self.swiglu_linear_clamp_min,
                    max=self.swiglu_linear_clamp_max,
                )
                gate = torch.clamp(gate, max=self.swiglu_gate_clamp_max)
            return self.proj(linear * F.silu(gate))
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class MoEMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_mult: int,
        activation_mode: str,
        swiglu_clamp_enabled: bool,
        swiglu_linear_clamp_min: float,
        swiglu_linear_clamp_max: float,
        swiglu_gate_clamp_max: float,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("MOE_NUM_EXPERTS must be >= 2 for MoE")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("MOE_TOP_K must be in [1, MOE_NUM_EXPERTS]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = CastedLinear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                MLP(
                    dim,
                    mlp_mult,
                    activation_mode,
                    swiglu_clamp_enabled,
                    swiglu_linear_clamp_min,
                    swiglu_linear_clamp_max,
                    swiglu_gate_clamp_max,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        router_logits = self.router(x).float()
        top_values, top_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1).to(dtype=x.dtype)
        result = torch.zeros_like(x)
        # This is a real routed MoE path. It computes only the selected expert
        # token subsets, which keeps memory use closer to a production top-k MoE.
        for expert_index, expert in enumerate(self.experts):
            selected = top_indices == expert_index
            if not bool(selected.any()):
                continue
            token_mask = selected.any(dim=-1)
            expert_input = x[token_mask]
            expert_output = expert(expert_input)
            weights = (selected.to(dtype=x.dtype) * top_weights).sum(dim=-1)[token_mask]
            result[token_mask] += expert_output * weights[:, None]
        return result


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        attn_norm_mode: str,
        attn_norm_eps: float,
        sparse_attn_mode: str,
        sparse_attn_window: int,
        sparse_attn_global_tokens: int,
        activation_mode: str,
        swiglu_clamp_enabled: bool,
        swiglu_linear_clamp_min: float,
        swiglu_linear_clamp_max: float,
        swiglu_gate_clamp_max: float,
        parallel_residual: bool,
        moe_num_experts: int,
        moe_top_k: int,
    ):
        super().__init__()
        self.parallel_residual = parallel_residual
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            attn_norm_mode,
            attn_norm_eps,
            sparse_attn_mode,
            sparse_attn_window,
            sparse_attn_global_tokens,
        )
        if moe_num_experts > 1:
            self.mlp = MoEMLP(
                dim,
                mlp_mult,
                activation_mode,
                swiglu_clamp_enabled,
                swiglu_linear_clamp_min,
                swiglu_linear_clamp_max,
                swiglu_gate_clamp_max,
                moe_num_experts,
                moe_top_k,
            )
        else:
            self.mlp = MLP(
                dim,
                mlp_mult,
                activation_mode,
                swiglu_clamp_enabled,
                swiglu_linear_clamp_min,
                swiglu_linear_clamp_max,
                swiglu_gate_clamp_max,
            )
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        if self.parallel_residual:
            mlp_out = self.mlp(self.mlp_norm(x))
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
            return x
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


def parse_layer_order(spec: str, default: list[int], num_layers: int, label: str) -> list[int]:
    if not spec:
        return default
    tokens = [token.strip() for token in spec.split(",")]
    if any(not token for token in tokens):
        raise ValueError(f"{label} contains an empty layer index: {spec!r}")
    order = [int(token) for token in tokens]
    for index in order:
        if index < 0 or index >= num_layers:
            raise ValueError(f"{label} index {index} is out of range for NUM_LAYERS={num_layers}")
    return order


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        attn_norm_mode: str,
        attn_norm_eps: float,
        sparse_attn_mode: str,
        sparse_attn_window: int,
        sparse_attn_global_tokens: int,
        activation_mode: str,
        swiglu_clamp_enabled: bool,
        swiglu_linear_clamp_min: float,
        swiglu_linear_clamp_max: float,
        swiglu_gate_clamp_max: float,
        parallel_residual: bool,
        encoder_layer_order: str,
        decoder_layer_order: str,
        moe_num_experts: int,
        moe_top_k: int,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        default_num_encoder_layers = num_layers // 2
        default_encoder_layer_order = list(range(default_num_encoder_layers))
        default_decoder_layer_order = list(range(default_num_encoder_layers, num_layers))
        self.encoder_layer_order = parse_layer_order(
            encoder_layer_order,
            default_encoder_layer_order,
            num_layers,
            "ENCODER_LAYER_ORDER",
        )
        self.decoder_layer_order = parse_layer_order(
            decoder_layer_order,
            default_decoder_layer_order,
            num_layers,
            "DECODER_LAYER_ORDER",
        )
        self.num_skip_weights = min(len(self.encoder_layer_order), len(self.decoder_layer_order))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    attn_norm_mode,
                    attn_norm_eps,
                    sparse_attn_mode,
                    sparse_attn_window,
                    sparse_attn_global_tokens,
                    activation_mode,
                    swiglu_clamp_enabled,
                    swiglu_linear_clamp_min,
                    swiglu_linear_clamp_max,
                    swiglu_gate_clamp_max,
                    parallel_residual,
                    moe_num_experts,
                    moe_top_k,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for block_index in self.encoder_layer_order:
            x = self.blocks[block_index](x, x0)
            skips.append(x)
        for i, block_index in enumerate(self.decoder_layer_order):
            if i < self.num_skip_weights and skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[block_index](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
