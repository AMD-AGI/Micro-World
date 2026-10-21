# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

# Fused elementwise kernels for the AdaLN-style modulation pattern used by
# WanAttentionBlock / CausalWanAttentionBlock:
#
#     temp_x = norm(x) * (1 + scale) + shift     -> modulate(norm_x, shift, scale)
#     x      = x + y * gate                      -> gated_residual(x, y, gate)
#
# Profiling (rocprofv3) of streaming inference showed these two patterns are
# each launched ~2 elementwise HIP kernels *per call*, and they are called
# 4 times per transformer block (2x modulate + 2x gated_residual) across
# num_layers=30 blocks * chunks -> thousands of tiny kernel launches whose
# combined *launch overhead* (not compute) dominates a meaningful slice of
# step time. This module fuses each pattern into a single kernel call.
#
# Safety / rollback design:
#   - Default backend ("eager") is *exactly* the original inline expression,
#     just moved into a function. Numerically and performance-wise it is a
#     no-op change versus the pre-fusion code.
#   - The backend is selected once via the WAN_ADALN_FUSION env var:
#         WAN_ADALN_FUSION=eager    (default, always safe)
#         WAN_ADALN_FUSION=script   torch.jit.script fusion
#     No source changes are needed to switch between them.
#   - Any failure to build/run the script backend is caught, logged once, and
#     the module permanently falls back to eager for the rest of the process.
#     A broken optimization path can therefore never crash a run.
#   - Set WAN_ADALN_FUSION_SELFCHECK=1 to additionally cross-check the
#     selected backend against the eager reference with torch.allclose on
#     every call and log the max abs/rel diff.
#
# Note on training: the script backend's autograd-generated backward is
# slower than eager's, and the launch-overhead win only exists for the small
# tensors of chunked streaming inference. Leave these switches unset when
# training (see examples/wan2.2/verify_training_compat.py).

import itertools
import os
import warnings
from collections import OrderedDict

import torch

_VALID_MODES = ("eager", "script")
_FUSION_MODE = os.environ.get("WAN_ADALN_FUSION", "eager").strip().lower()
if _FUSION_MODE not in _VALID_MODES:
    warnings.warn(
        f"[fused_ops] Unknown WAN_ADALN_FUSION={_FUSION_MODE!r}, "
        f"expected one of {_VALID_MODES}. Falling back to 'eager'.")
    _FUSION_MODE = "eager"

_SELFCHECK = os.environ.get("WAN_ADALN_FUSION_SELFCHECK", "0").strip() == "1"

_warned_once = set()


def _warn_once(key, msg):
    if key not in _warned_once:
        _warned_once.add(key)
        warnings.warn(msg)


# ---------------------------------------------------------------------------
# Backend 0: eager (baseline, always correct, always available)
# ---------------------------------------------------------------------------

def _modulate_eager(x, shift, scale):
    return x * (1 + scale) + shift


def _gated_residual_eager(x, y, gate):
    return x + y * gate


# ---------------------------------------------------------------------------
# Backend 1: TorchScript
# ---------------------------------------------------------------------------

try:
    _modulate_script = torch.jit.script(_modulate_eager)
    _gated_residual_script = torch.jit.script(_gated_residual_eager)
    _SCRIPT_OK = True
except Exception as exc:  # pragma: no cover - defensive
    _SCRIPT_OK = False
    _script_build_error = exc


# ---------------------------------------------------------------------------
# Dispatch with automatic, permanent-for-process-lifetime fallback to eager
# ---------------------------------------------------------------------------

_active_mode = "eager"  # updated below once we know the requested backend is usable

if _FUSION_MODE == "script" and not _SCRIPT_OK:
    _warn_once(
        "script_build",
        f"[fused_ops] torch.jit.script build failed ({_script_build_error!r}); using eager.")
else:
    _active_mode = _FUSION_MODE


def _fallback_to_eager(op_name, exc):
    global _active_mode
    _warn_once(
        f"runtime_fail_{op_name}",
        f"[fused_ops] backend={_active_mode!r} failed at runtime for {op_name}() "
        f"({exc!r}); permanently falling back to eager for this process.")
    _active_mode = "eager"


def _run_backend(op_name, eager_fn, script_fn, args):
    mode = _active_mode
    if mode == "eager":
        return eager_fn(*args)
    try:
        out = script_fn(*args)
    except Exception as exc:
        _fallback_to_eager(op_name, exc)
        return eager_fn(*args)

    if _SELFCHECK:
        ref = eager_fn(*args)
        if not torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2):
            diff = (out.float() - ref.float()).abs().max().item()
            _warn_once(
                f"selfcheck_{op_name}",
                f"[fused_ops] SELFCHECK MISMATCH backend={mode!r} {op_name}(): "
                f"max abs diff={diff:.4g} exceeds tolerance; falling back to eager.")
            _fallback_to_eager(op_name, RuntimeError("selfcheck mismatch"))
            return ref
    return out


def modulate(x, shift, scale):
    """out = x * (1 + scale) + shift, with shift/scale broadcastable to x's shape."""
    return _run_backend(
        "modulate", _modulate_eager,
        _modulate_script if _SCRIPT_OK else None, (x, shift, scale))


def gated_residual(x, y, gate):
    """out = x + y * gate, with gate broadcastable to x's shape."""
    return _run_backend(
        "gated_residual", _gated_residual_eager,
        _gated_residual_script if _SCRIPT_OK else None, (x, y, gate))


print(
    f"[fused_ops] AdaLN modulate/gated_residual backend = {_active_mode!r} "
    f"(requested={_FUSION_MODE!r}, selfcheck={_SELFCHECK}). "
    f"Set WAN_ADALN_FUSION=eager|script to change, "
    f"WAN_ADALN_FUSION_SELFCHECK=1 to cross-check against eager at runtime."
)


# ===========================================================================
# RMSNorm fusion (WanRMSNorm._norm(x) * weight)
#
#   out = x * rsqrt(mean(x**2, dim=-1) + eps).to(x.dtype) * weight
#
# Profiling showed this single call (used for norm_q/norm_k in every
# self-attention call) is launched as ~5 separate elementwise/reduction
# kernels (pow, mean-reduce, cast, mul, mul) purely because the eager
# PyTorch expression has no dedicated fused ATen op (unlike nn.LayerNorm,
# which already has a single native fused kernel and is therefore *not*
# touched here).
#
# Safety / rollback: identical pattern to the AdaLN section above.
#   WAN_RMSNORM_FUSION=eager|script (default: eager, exact no-op versus the
#   original inline code)
#   WAN_RMSNORM_FUSION_SELFCHECK=1 to cross-check against eager at runtime.
# ===========================================================================

_RMSNORM_MODE = os.environ.get("WAN_RMSNORM_FUSION", "eager").strip().lower()
if _RMSNORM_MODE not in _VALID_MODES:
    warnings.warn(
        f"[fused_ops] Unknown WAN_RMSNORM_FUSION={_RMSNORM_MODE!r}, "
        f"expected one of {_VALID_MODES}. Falling back to 'eager'.")
    _RMSNORM_MODE = "eager"

_RMSNORM_SELFCHECK = os.environ.get("WAN_RMSNORM_FUSION_SELFCHECK", "0").strip() == "1"


def _rms_norm_eager(x, weight, eps: float):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype) * weight


try:
    _rms_norm_script = torch.jit.script(_rms_norm_eager)
    _RMSNORM_SCRIPT_OK = True
except Exception as exc:  # pragma: no cover - defensive
    _RMSNORM_SCRIPT_OK = False
    _rmsnorm_script_build_error = exc


_rmsnorm_active_mode = "eager"

if _RMSNORM_MODE == "script" and not _RMSNORM_SCRIPT_OK:
    _warn_once(
        "rmsnorm_script_build",
        f"[fused_ops] RMSNorm torch.jit.script build failed "
        f"({_rmsnorm_script_build_error!r}); using eager.")
else:
    _rmsnorm_active_mode = _RMSNORM_MODE


def _rmsnorm_fallback_to_eager(exc):
    global _rmsnorm_active_mode
    _warn_once(
        "rmsnorm_runtime_fail",
        f"[fused_ops] RMSNorm backend={_rmsnorm_active_mode!r} failed at runtime "
        f"({exc!r}); permanently falling back to eager for this process.")
    _rmsnorm_active_mode = "eager"


def rms_norm(x, weight, eps: float = 1e-5):
    """out = x * rsqrt(mean(x**2, dim=-1) + eps).to(x.dtype) * weight"""
    mode = _rmsnorm_active_mode
    if mode == "eager":
        return _rms_norm_eager(x, weight, eps)
    try:
        out = _rms_norm_script(x, weight, eps)
    except Exception as exc:
        _rmsnorm_fallback_to_eager(exc)
        return _rms_norm_eager(x, weight, eps)

    if _RMSNORM_SELFCHECK:
        ref = _rms_norm_eager(x, weight, eps)
        if not torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2):
            diff = (out.float() - ref.float()).abs().max().item()
            _warn_once(
                "rmsnorm_selfcheck",
                f"[fused_ops] SELFCHECK MISMATCH backend={mode!r} rms_norm(): "
                f"max abs diff={diff:.4g} exceeds tolerance; falling back to eager.")
            _rmsnorm_fallback_to_eager(RuntimeError("selfcheck mismatch"))
            return ref
    return out


def get_fusion_status():
    return {
        "adaln": {
            "requested_mode": _FUSION_MODE,
            "active_mode": _active_mode,
            "selfcheck": _SELFCHECK,
            "script_available": _SCRIPT_OK,
        },
        "rmsnorm": {
            "requested_mode": _RMSNORM_MODE,
            "active_mode": _rmsnorm_active_mode,
            "selfcheck": _RMSNORM_SELFCHECK,
            "script_available": _RMSNORM_SCRIPT_OK,
        },
    }


print(
    f"[fused_ops] RMSNorm backend = {_rmsnorm_active_mode!r} "
    f"(requested={_RMSNORM_MODE!r}, selfcheck={_RMSNORM_SELFCHECK}). "
    f"Set WAN_RMSNORM_FUSION=eager|script to change, "
    f"WAN_RMSNORM_FUSION_SELFCHECK=1 to cross-check against eager at runtime."
)


# ===========================================================================
# QKV projection fusion (CausalWanSelfAttention.qkv_fn)
#
# The original code runs self.q(x), self.k(x), self.v(x) as three separate
# nn.Linear(dim, dim) calls on the *same* input x. Profiling (rocprofv3,
# steady-state chunk) showed the transformer's GEMMs are tiny-M (chunk_size=3
# frames -> a few thousand tokens at most) and therefore launch/dispatch-
# overhead bound, not compute bound (~43us average per GEMM kernel, in line
# with the ROCm kernel-launch floor). Concatenating the three [dim, dim]
# weight matrices into one [3*dim, dim] matrix and issuing a single
# F.linear(x, W_qkv, b_qkv) call is mathematically IDENTICAL (same GEMM
# arithmetic, just computed as one wider matmul instead of three separate
# ones) and cuts 3 GEMM kernel launches down to 1 per self-attention call
# (q/k/v), i.e. -66% of the self-attn projection kernel count, with strictly
# more work per launch (better occupancy on this dispatch-bound shape).
#
# LoRA is already merged into q/k/v.weight *before* inference starts (see
# examples/wan2.2/benchmark_streaming_chunk_time.py::merge_lora), so at
# inference time q/k/v.weight are plain static tensors -- concatenating them
# once (cached, invalidated on weight identity/version change) is safe and
# captures any LoRA-merged weights automatically.
#
# Safety / rollback: same toggle pattern as the sections above.
#   WAN_QKV_FUSION=1 to enable (default: 0, i.e. the original 3 separate
#   nn.Linear calls -- exact no-op versus pre-fusion code).
#   WAN_QKV_FUSION_SELFCHECK=1 to cross-check against the 3-separate-calls
#   reference with torch.allclose on every call.
# ===========================================================================

QKV_FUSION = os.environ.get("WAN_QKV_FUSION", "0").strip() == "1"
QKV_FUSION_SELFCHECK = os.environ.get("WAN_QKV_FUSION_SELFCHECK", "0").strip() == "1"

_fused_linear_cache = {}  # cache_key -> (weight, bias_or_None, tuple(weight_tensor identities+versions))


def _fingerprint_equal(a, b):
    """Compare two weight fingerprints. Tensors are matched by *identity*:
    `==` on two distinct tensors would return an elementwise mask, and the
    cache keys embed id(module), which the allocator may hand out again to a
    different module once the first one is freed."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, tuple) or isinstance(y, tuple):
            if not (isinstance(x, tuple) and isinstance(y, tuple) and _fingerprint_equal(x, y)):
                return False
        elif torch.is_tensor(x) or torch.is_tensor(y):
            if x is not y:
                return False
        elif x != y:
            return False
    return True


def _get_fused_linear_weight(cache_key, linears):
    """Concatenate N nn.Linear layers (same in_features, sharing the same
    input) into one [sum(out_features), in_features] weight (+ bias), cached
    and invalidated on any of their weight tensors' identity/version
    changing (PyTorch bumps Tensor._version on every in-place write, so this
    catches e.g. a future .load_state_dict() or LoRA re-merge for free)."""
    fingerprint = tuple((l.weight, l.weight._version) for l in linears)
    entry = _fused_linear_cache.get(cache_key)
    if entry is not None and _fingerprint_equal(entry[2], fingerprint):
        return entry[0], entry[1]
    w = torch.cat([l.weight for l in linears], dim=0)
    if linears[0].bias is not None:
        b = torch.cat([l.bias for l in linears], dim=0)
    else:
        b = None
    _fused_linear_cache[cache_key] = (w, b, fingerprint)
    return w, b


def fused_qkv(attn_module, x):
    """Return (q_raw, k_raw, v_raw) = (self.q(x), self.k(x), self.v(x)) computed
    via a single concatenated-weight matmul instead of three separate calls.
    Each output has the same shape [..., dim] as the corresponding original
    call (norm_q/norm_k/view are applied by the caller, unchanged)."""
    w, b = _get_fused_linear_weight(
        ("qkv", id(attn_module)), [attn_module.q, attn_module.k, attn_module.v])
    dim = attn_module.dim
    out = torch.nn.functional.linear(x, w, b)
    return out[..., :dim], out[..., dim:2 * dim], out[..., 2 * dim:]


def fused_kv(attn_module, context, k_attr="k", v_attr="v"):
    """Return (k_raw, v_raw) = (self.k(context), self.v(context)) (or the
    k_img/v_img pair when k_attr/v_attr override the default names) computed
    via a single concatenated-weight matmul. Used for cross-attention, where
    k and v are both linear projections of the *same* context tensor."""
    k_layer = getattr(attn_module, k_attr)
    v_layer = getattr(attn_module, v_attr)
    w, b = _get_fused_linear_weight(
        ("kv", id(attn_module), k_attr, v_attr), [k_layer, v_layer])
    dim = attn_module.dim
    out = torch.nn.functional.linear(context, w, b)
    return out[..., :dim], out[..., dim:]


print(
    f"[fused_ops] QKV/KV projection fusion = {QKV_FUSION} "
    f"(selfcheck={QKV_FUSION_SELFCHECK}). "
    f"Set WAN_QKV_FUSION=1 to fuse self-attn q/k/v and cross-attn k/v into "
    f"single GEMMs (default: 0, original separate nn.Linear calls)."
)


# ===========================================================================
# Prompt-context cache across denoising steps and chunks (WAN_CTX_CACHE)
# ===========================================================================
# In streaming inference the text prompt is encoded once, before the chunk
# loop, and the very same `prompt_embeds` tensor is passed to every
# transformer.forward() call. Everything downstream of it that does *not*
# depend on x / t / the kv-caches is therefore recomputed identically on
# every call:
#
#   1. the forward() preamble: zero-pad to text_len -> cast -> text_embedding
#      (2 GEMMs + GELU),  4 denoising steps per chunk;
#   2. per cross-attention layer: context.to(dtype) -> k/v projection ->
#      norm_k -> view,  30 layers x 4 steps = 120 times per chunk.
#
# Memoizing both is bit-exact (same kernels, same inputs => same outputs),
# and removes ~128 GEMM launches and ~120 RMSNorm launches per chunk.
#
# Correctness / invalidation:
#   - The embedded context is keyed on the *identity* (data_ptr + shape +
#     dtype + device + Tensor._version) of the raw `context` argument, and a
#     strong reference to it is kept so its address cannot be recycled by a
#     different tensor. A new prompt therefore always misses.
#   - Only the tensor produced by embed_context() is tagged as cacheable, so
#     derived tensors (e.g. the i2v `torch.concat([context_clip, context])`,
#     which depends on clip_fea) transparently fall back to recomputation.
#   - Cached k/v additionally carry a fingerprint of the projection/norm
#     weights, so a post-load LoRA re-merge or load_state_dict invalidates
#     them instead of silently serving stale activations.
#
# Cost: 30 layers x (k, v) x [1, text_len, dim] bf16 ~= 180 MB of values per
# distinct prompt, ~280 MB of held storage in the default (non-densified)
# mode, doubled when CFG runs a separate negative prompt.
#
# Safety / rollback: same toggle pattern as the sections above.
#   WAN_CTX_CACHE=1 to enable (default: 0, i.e. recompute everything, an
#   exact no-op versus pre-cache code).
#   WAN_CTX_CACHE_SELFCHECK=1 to recompute on every hit and assert the cached
#   value is bit-identical (torch.equal) before using it.
#   WAN_CTX_CACHE_MAX=N caps the number of distinct prompts kept (default 2,
#   i.e. one prompt's cond + uncond; a new prompt evicts the oldest).
#   WAN_CTX_CACHE_WARMUP=N discards the first N computations of each entry
#   before caching it (default 1, see the note on TorchScript warmup below).
#   WAN_CTX_CACHE_CONTIGUOUS=1 to densify the cached values, which drops the
#   footprint to ~180 MB by not pinning the fused k/v buffers. Verified
#   bit-identical on the flash-attn path, but left opt-in because it hands
#   attention() a different memory layout than the uncached code does.
# ===========================================================================

CTX_CACHE = os.environ.get("WAN_CTX_CACHE", "0").strip() == "1"
CTX_CACHE_SELFCHECK = os.environ.get("WAN_CTX_CACHE_SELFCHECK", "0").strip() == "1"
CTX_CACHE_CONTIGUOUS = os.environ.get("WAN_CTX_CACHE_CONTIGUOUS", "0").strip() == "1"


def _ctx_int_env(name, default):
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


_CTX_CACHE_MAX = max(1, _ctx_int_env("WAN_CTX_CACHE_MAX", 2))
_CTX_CACHE_WARMUP = _ctx_int_env("WAN_CTX_CACHE_WARMUP", 1)

_ctx_embed_cache = OrderedDict()  # embed_key -> _CachedContext
_ctx_kv_cache = {}  # kv_key -> (k, v, weight_fingerprint)
_ctx_kv_warmup = {}  # kv_key -> computations still to discard before caching
_ctx_uid_counter = itertools.count(1)
ctx_cache_stats = {"embed_hit": 0, "embed_miss": 0, "kv_hit": 0, "kv_miss": 0,
                   "kv_bypass": 0, "evicted": 0}


class _CachedContext:
    __slots__ = ("uid", "raw_refs", "embedded")

    def __init__(self, uid, raw_refs, embedded):
        self.uid = uid
        self.raw_refs = raw_refs  # keeps the source tensors alive => data_ptr stays unique
        self.embedded = embedded


def _context_input_key(context):
    """Identity key for the raw `context` argument of WanModel.forward, plus
    the list of tensors it is made of (kept alive by the cache entry).
    Returns (None, None) for anything that is not a tensor / list of
    tensors, which disables caching for that call."""
    tensors = list(context) if isinstance(context, (list, tuple)) else [context]
    key = []
    for t in tensors:
        if not torch.is_tensor(t):
            return None, None
        key.append((t.data_ptr(), tuple(t.shape), t.dtype, str(t.device), t._version))
    return tuple(key), tensors


def _drop_kv_entries(uid):
    for d in (_ctx_kv_cache, _ctx_kv_warmup):
        for k in [k for k in d if k[0] == uid]:
            del d[k]


def clear_context_cache():
    """Drop every cached prompt context and its per-layer k/v (frees the
    ~180 MB/prompt of cached activations). Safe to call at any time."""
    _ctx_embed_cache.clear()
    _ctx_kv_cache.clear()
    _ctx_kv_warmup.clear()


def get_context_cache_status():
    return {
        "enabled": CTX_CACHE,
        "selfcheck": CTX_CACHE_SELFCHECK,
        "prompts_cached": len(_ctx_embed_cache),
        "kv_entries": len(_ctx_kv_cache),
        "kv_warming": len(_ctx_kv_warmup),
        "kv_bytes": _ctx_kv_bytes(),
        **ctx_cache_stats,
    }


def _ctx_kv_bytes():
    """Bytes actually held by the cached k/v. Deduplicated by storage, since
    without WAN_CTX_CACHE_CONTIGUOUS the cached v is a view into the (2x
    larger) fused k/v buffer, which it keeps alive in full."""
    storages = {}
    for k, v, _ in _ctx_kv_cache.values():
        for t in (k, v):
            s = t.untyped_storage()
            storages[s.data_ptr()] = s.nbytes()
    return sum(storages.values())


def embed_context(model, context):
    """Run WanModel.forward's prompt-embedding preamble -- zero-pad each
    sequence to model.text_len, cast to the text_embedding dtype, then apply
    model.text_embedding -- memoized on the identity of `context`.

    The returned tensor is tagged so the cross-attention layers below can key
    their k/v cache on it. With WAN_CTX_CACHE=0 this is exactly the original
    inline code path."""

    def _compute():
        c = torch.stack([
            torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
            for u in context
        ])
        c = c.to(dtype=next(model.text_embedding.parameters()).dtype)
        return model.text_embedding(c)

    if not CTX_CACHE:
        return _compute()

    key, refs = _context_input_key(context)
    if key is None:
        return _compute()
    key = (id(model),) + key

    entry = _ctx_embed_cache.get(key)
    if entry is not None:
        _ctx_embed_cache.move_to_end(key)
        ctx_cache_stats["embed_hit"] += 1
        if CTX_CACHE_SELFCHECK:
            ref = _compute()
            if not torch.equal(ref, entry.embedded):
                _warn_once(
                    "ctx_cache_embed_selfcheck",
                    "[fused_ops] SELFCHECK MISMATCH cached text_embedding(context): "
                    "recomputing instead of using the cached value.")
                return ref
        return entry.embedded

    ctx_cache_stats["embed_miss"] += 1
    out = _compute()
    entry = _CachedContext(next(_ctx_uid_counter), refs, out)
    out._wan_ctx_uid = entry.uid
    _ctx_embed_cache[key] = entry
    while len(_ctx_embed_cache) > _CTX_CACHE_MAX:
        _, evicted = _ctx_embed_cache.popitem(last=False)
        _drop_kv_entries(evicted.uid)
        ctx_cache_stats["evicted"] += 1
    return out


def _kv_weight_fingerprint(k_layer, v_layer, norm):
    w = getattr(norm, "weight", None)
    return (k_layer.weight, k_layer.weight._version,
            v_layer.weight, v_layer.weight._version,
            w, None if w is None else w._version)


def _cross_attn_kv_compute(attn_module, context, dtype, b, n, d, k_attr, v_attr, norm_attr):
    """The original inline cross-attention key/value path: cast the context,
    project it to k/v, RMSNorm the keys, and reshape both to [B, L, N, D]."""
    context_dt = context.to(dtype)
    k_layer = getattr(attn_module, k_attr)
    v_layer = getattr(attn_module, v_attr)
    if QKV_FUSION:
        k_raw, v_raw = fused_kv(attn_module, context_dt, k_attr, v_attr)
        if QKV_FUSION_SELFCHECK:
            ref_k, ref_v = k_layer(context_dt), v_layer(context_dt)
            if not (torch.allclose(k_raw.float(), ref_k.float(), rtol=1e-2, atol=1e-2)
                    and torch.allclose(v_raw.float(), ref_v.float(), rtol=1e-2, atol=1e-2)):
                _warn_once(
                    "kv_fusion_selfcheck",
                    "[fused_ops] SELFCHECK MISMATCH fused_kv() (cross-attn): "
                    "falling back to eager k/v.")
                k_raw, v_raw = ref_k, ref_v
    else:
        k_raw, v_raw = k_layer(context_dt), v_layer(context_dt)
    k = getattr(attn_module, norm_attr)(k_raw).view(b, -1, n, d)
    v = v_raw.view(b, -1, n, d)
    return k, v


def cross_attn_kv(attn_module, context, dtype, b, n, d,
                  k_attr="k", v_attr="v", norm_attr="norm_k"):
    """Cross-attention k/v for `context`, served from the prompt cache when
    `context` came from embed_context() and WAN_CTX_CACHE=1, otherwise
    computed exactly as before."""
    uid = getattr(context, "_wan_ctx_uid", None) if CTX_CACHE else None
    if uid is None:
        if CTX_CACHE:
            ctx_cache_stats["kv_bypass"] += 1
        return _cross_attn_kv_compute(
            attn_module, context, dtype, b, n, d, k_attr, v_attr, norm_attr)

    key = (uid, id(attn_module), k_attr, v_attr, dtype, b)
    fingerprint = _kv_weight_fingerprint(
        getattr(attn_module, k_attr), getattr(attn_module, v_attr),
        getattr(attn_module, norm_attr))
    entry = _ctx_kv_cache.get(key)
    if entry is not None and _fingerprint_equal(entry[2], fingerprint):
        ctx_cache_stats["kv_hit"] += 1
        if CTX_CACHE_SELFCHECK:
            ref_k, ref_v = _cross_attn_kv_compute(
                attn_module, context, dtype, b, n, d, k_attr, v_attr, norm_attr)
            if not (torch.equal(entry[0], ref_k) and torch.equal(entry[1], ref_v)):
                _warn_once(
                    "ctx_cache_kv_selfcheck",
                    "[fused_ops] SELFCHECK MISMATCH cached cross-attn k/v: "
                    "recomputing instead of using the cached value.")
                return ref_k, ref_v
        return entry[0], entry[1]

    ctx_cache_stats["kv_miss"] += 1
    k, v = _cross_attn_kv_compute(
        attn_module, context, dtype, b, n, d, k_attr, v_attr, norm_attr)
    # The TorchScript RMSNorm backend interprets its graph on the first call
    # and only then swaps in the fused kernel, whose rounding differs by ~1
    # bf16 ULP. Discarding the first few results keeps the cache pinned to the
    # steady-state value the uncached model would have produced.
    pending = _ctx_kv_warmup.get(key, _CTX_CACHE_WARMUP)
    if pending > 0:
        _ctx_kv_warmup[key] = pending - 1
        return k, v
    if CTX_CACHE_CONTIGUOUS:
        # With QKV_FUSION on, v is a strided view into the [.., 2*dim] fused
        # output; densifying it saves ~100 MB of pinned buffers and lets
        # attention() skip its internal copy. NOT bit-exact: the attention
        # kernel picks a different path for contiguous inputs, so this is
        # opt-in rather than the default.
        k, v = k.contiguous(), v.contiguous()
    _ctx_kv_warmup.pop(key, None)
    _ctx_kv_cache[key] = (k, v, fingerprint)
    return k, v


print(
    f"[fused_ops] Prompt-context cache = {CTX_CACHE} "
    f"(selfcheck={CTX_CACHE_SELFCHECK}, max_prompts={_CTX_CACHE_MAX}). "
    f"Set WAN_CTX_CACHE=1 to reuse text_embedding(context) and the "
    f"cross-attn k/v projections across denoising steps and chunks "
    f"(default: 0, recompute every forward)."
)
