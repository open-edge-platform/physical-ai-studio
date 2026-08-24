# Copyright 2024-2025 The Robbyant Team Authors.
# Copyright 2026 The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Attention backends and rotary embeddings for the LingBot-VA Wan2.2 backbone.

Three self-attention backends are available:

- ``torch`` — PyTorch SDPA. Always available; the inference default.
- ``flashattn`` — optional ``flash_attn`` kernels.
- ``flex`` — PyTorch flex-attention, used only for training because the dual-stream
  flow-matching loss needs block-causal / window / noise-vs-clean masks.

:class:`WanAttention` also owns the paged KV cache that makes autoregressive streaming
inference possible: keys/values for previously generated chunks are kept in a fixed-size
pool, and slots are evicted oldest-first once the attention window is full.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def custom_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Scaled-dot-product attention operating on ``(B, S, H, D)`` tensors.

    Args:
        query: Query tensor of shape ``[B, S_q, H, D]``.
        key: Key tensor of shape ``[B, S_kv, H, D]``.
        value: Value tensor of shape ``[B, S_kv, H, D]``.

    Returns:
        Attention output of shape ``[B, S_q, H, D]``.
    """
    out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    return out.transpose(1, 2)


def load_flash_attn_func() -> Callable[..., torch.Tensor]:
    """Import ``flash_attn_func`` from whichever flash-attention package is installed.

    Returns:
        The ``flash_attn_func`` callable.

    Raises:
        ImportError: If neither ``flash_attn_interface`` nor ``flash_attn`` is installed.
    """
    try:
        from flash_attn_interface import flash_attn_func  # noqa: PLC0415  # pyrefly: ignore[missing-import]
    except ImportError:
        try:
            from flash_attn import flash_attn_func  # noqa: PLC0415  # pyrefly: ignore[missing-import]
        except ImportError as e:
            msg = (
                "attn_mode='flashattn' requires the `flash_attn` package, which is not installed. "
                "Install it, or use attn_mode='torch' (the default)."
            )
            raise ImportError(msg) from e
    return flash_attn_func


class FlexAttnFunc(nn.Module):
    """Flex-attention backend (training only, ``attn_mode='flex'``).

    Builds the block-causal / window / noise-vs-clean masks used by the dual-stream
    flow-matching training pass. The flex-attention APIs and their ``torch.compile``
    wrappers are imported lazily so importing this module never requires a
    flex-attention-capable PyTorch build.

    The block masks are class-level state: :meth:`init_mask` is called once per training
    step (from the transformer's training forward) and every attention module in the
    stack then reads the same masks.
    """

    flex_attn: Any = None
    compiled_create_block_mask: Any = None
    attention_mask: Any = None
    cross_attention_mask: Any = None

    def __init__(self, *, is_cross: bool = False) -> None:
        """Initialize the backend.

        Args:
            is_cross: Whether this instance serves cross-attention (text conditioning)
                rather than self-attention.
        """
        super().__init__()
        self.is_cross = is_cross

    @classmethod
    def _ensure_compiled(cls) -> None:
        """Import and compile the flex-attention entry points once."""
        if cls.flex_attn is None:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa: PLC0415

            cls.flex_attn = torch.compile(flex_attention, dynamic=True)
            cls.compiled_create_block_mask = torch.compile(create_block_mask)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Run masked flex attention over the packed dual-stream sequence.

        Args:
            query: Query tensor of shape ``[1, S, H, D]``.
            key: Key tensor of shape ``[1, S, H, D]``.
            value: Value tensor of shape ``[1, S, H, D]``.
            dtype: Half-precision dtype to cast to.

        Returns:
            Attention output of shape ``[1, S, H, D]``.

        Raises:
            ValueError: If ``dtype`` is not a half-precision dtype.
        """
        self._ensure_compiled()
        if dtype not in _HALF_DTYPES:
            msg = f"Flex attention requires a half-precision dtype, got {dtype}."
            raise ValueError(msg)

        def half(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype in _HALF_DTYPES else x.to(dtype)

        q_varlen = half(rearrange(query[0], "s n d -> 1 n s d"))
        k_varlen = half(rearrange(key[0], "s n d -> 1 n s d"))
        v_varlen = half(rearrange(value[0], "s n d -> 1 n s d"))
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask

        x_out = FlexAttnFunc.flex_attn(  # type: ignore[misc]
            q_varlen,
            k_varlen,
            v_varlen,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        )
        return rearrange(x_out, "b n s d -> b s n d")

    @staticmethod
    @torch.no_grad()
    def init_mask(  # noqa: PLR0914
        latent_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        padded_length: int,
        chunk_size: int,
        window_size: int,
        patch_size: tuple[int, int, int],
        device: torch.device,
        text_seq_len: int = 512,
    ) -> None:
        """Build the block-causal self-attention and cross-attention masks.

        The packed sequence is ``[noisy latents, clean latents, noisy actions, clean
        actions, padding]``. Frames are grouped into blocks of ``chunk_size`` frames,
        with the latent and action streams of the same block interleaved so actions
        attend to the latents of their own block. Within that structure, clean tokens
        attend block-causally, noisy tokens attend to strictly earlier clean tokens and
        to noisy tokens of their own block, and every pair is restricted to
        ``window_size`` blocks.

        Args:
            latent_shape: Shape of the noisy latent stream ``[B, C, F, H, W]``.
            action_shape: Shape of the noisy action stream ``[B, C, F, N, 1]``.
            padded_length: Number of padding tokens appended to reach a multiple of 128.
            chunk_size: Number of latent frames per autoregressive block.
            window_size: Maximum block distance two tokens may attend across.
            patch_size: Latent patch size ``(t, h, w)``.
            device: Device to build the masks on.
            text_seq_len: Length of the text-conditioning sequence.
        """
        FlexAttnFunc._ensure_compiled()
        torch._inductor.config.realize_opcount_threshold = 100  # noqa: SLF001
        b, _, l_f, l_h, l_w = latent_shape
        _, _, a_f, a_h, a_w = action_shape

        latent_seq_id = (
            torch.arange(b)[:, None, None, None]
            .expand(-1, l_f // patch_size[0], l_h // patch_size[1], l_w // patch_size[2])
            .flatten()
        )
        action_seq_id = torch.arange(b)[:, None, None, None].expand(-1, a_f, a_h, a_w).flatten()
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

        latent_frame_id = (
            torch.arange(l_f)[None, :, None, None]
            .expand(b, -1, l_h // patch_size[1], l_w // patch_size[2])[None]
            .flatten()
        )
        action_frame_id = torch.arange(a_f)[None, :, None, None].expand(b, -1, a_h, a_w)[None].flatten()
        frame_ids = torch.cat(
            [latent_frame_id // chunk_size * 2] * 2 + [action_frame_id // chunk_size * 2 + 1] * 2,
        )

        noise_ids = torch.cat([
            torch.zeros_like(latent_frame_id),
            torch.ones_like(latent_frame_id),
            torch.zeros_like(action_frame_id),
            torch.ones_like(action_frame_id),
        ])

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        mask_mod = FlexAttnFunc._get_mask_mod(
            seq_ids.long().to(device),
            frame_ids.long().to(device),
            noise_ids.long().to(device),
            window_size,
        )
        FlexAttnFunc.attention_mask = FlexAttnFunc.compiled_create_block_mask(  # type: ignore[misc]
            mask_mod,
            1,
            1,
            len(seq_ids),
            len(seq_ids),
            device=device,
            _compile=True,
        )

        text_seq_ids = torch.arange(b)[:, None].expand(-1, text_seq_len).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(
            seq_ids.long().to(device),
            text_seq_ids.long().to(device),
        )
        FlexAttnFunc.cross_attention_mask = FlexAttnFunc.compiled_create_block_mask(  # type: ignore[misc]
            mask_mod_cross,
            1,
            1,
            len(seq_ids),
            len(text_seq_ids),
            device=device,
            _compile=True,
        )

    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids: torch.Tensor, text_seq_ids: torch.Tensor) -> Callable[..., torch.Tensor]:
        """Build the cross-attention mask: each sample attends only to its own prompt.

        Returns:
            A flex-attention ``mask_mod`` callable.
        """

        def seq_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
            del b, h
            return (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >= 0) & (text_seq_ids[kv_idx] >= 0)

        return seq_mask

    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(
        seq_ids: torch.Tensor,
        frame_ids: torch.Tensor,
        noise_ids: torch.Tensor,
        window_size: int,
    ) -> Callable[..., torch.Tensor]:
        """Build the block-causal / window / noise-vs-clean self-attention mask.

        Returns:
            A flex-attention ``mask_mod`` callable.
        """
        from torch.nn.attention.flex_attention import and_masks, or_masks  # noqa: PLC0415

        def seq_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
            del b, h
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >= 0) & (seq_ids[kv_idx] >= 0)

        def block_causal_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            return frame_ids[kv_idx] <= frame_ids[q_idx]

        def block_causal_mask_exclude_self(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            return frame_ids[kv_idx] < frame_ids[q_idx]

        def block_self_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            return frame_ids[kv_idx] == frame_ids[q_idx]

        def clean2clean_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)

        def noise2clean_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)

        def noise2noise_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)

        def block_window_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
            window_size: int,
        ) -> torch.Tensor:
            del b, h
            return (frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size

        mask = or_masks(
            and_masks(clean2clean_mask, block_causal_mask),
            and_masks(noise2clean_mask, block_causal_mask_exclude_self),
            and_masks(noise2noise_mask, block_self_mask),
        )
        mask = and_masks(mask, seq_mask)
        return and_masks(mask, partial(block_window_mask, window_size=window_size))


class WanRotaryPosEmbed(nn.Module):
    """Rotary position embedding with separate frequency bases for frame / height / width.

    Args:
        attention_head_dim: Per-head dimension; split across the three axes.
        patch_size: Latent patch size ``(t, h, w)``.
        max_seq_len: Maximum sequence length (kept for parity with the upstream config).
        theta: Rotary base.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ) -> None:
        """Initialize the embedding and precompute the per-axis frequency bases."""
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        self.f_freqs_base, self.h_freqs_base, self.w_freqs_base = self._precompute_freqs_base()

    def _precompute_freqs_base(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute the inverse-frequency vectors for the three axes.

        Returns:
            Tuple of ``(frame, height, width)`` inverse-frequency tensors.
        """

        def base(dim: int) -> torch.Tensor:
            return 1.0 / (self.theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))

        return base(self.f_dim), base(self.h_dim), base(self.w_dim)

    def forward(self, grid_ids: torch.Tensor) -> torch.Tensor:
        """Map grid ids to complex rotary factors.

        Args:
            grid_ids: Tensor of shape ``[B, 4, S]`` holding the frame / height / width /
                stream ids of every token.

        Returns:
            Complex tensor of shape ``[B, S, head_dim // 2]``.
        """
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            return torch.polar(torch.ones_like(freqs), freqs)


class WanAttention(nn.Module):
    """Self/cross attention with a paged KV cache for autoregressive streaming inference.

    The KV cache lives only on self-attention modules (``cross_attention_dim_head is
    None``). Each forward writes its keys/values into free slots of a fixed-size pool and
    attends over every valid slot; when ``update_cache == 0`` the freshly written slots
    are released again, so speculative denoising steps do not pollute the cache.

    Args:
        dim: Model dimension.
        heads: Number of attention heads.
        dim_head: Per-head dimension.
        eps: Epsilon of the query/key RMS norms.
        dropout: Dropout probability on the output projection.
        cross_attention_dim_head: Per-head dimension of the cross-attention keys/values.
            ``None`` selects self-attention (and enables the KV cache).
        attn_mode: One of ``"torch"``, ``"flashattn"``, ``"flex"``.

    Raises:
        ValueError: If ``attn_mode`` is not a supported backend.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        cross_attention_dim_head: int | None = None,
        attn_mode: str = "torch",
    ) -> None:
        """Initialize the projections, norms and (for self-attention) the cache slot.

        Raises:
            ValueError: If ``attn_mode`` is not a supported backend.
        """
        super().__init__()
        if attn_mode == "torch":
            self.attn_op: Callable[..., torch.Tensor] = custom_sdpa
        elif attn_mode == "flashattn":
            self.attn_op = load_flash_attn_func()
        elif attn_mode == "flex":
            self.attn_op = FlexAttnFunc(is_cross=cross_attention_dim_head is not None)
        else:
            msg = f"Unsupported attention mode: {attn_mode}, only support 'torch', 'flashattn' and 'flex'"
            raise ValueError(msg)

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=True), nn.Dropout(dropout)])
        self.norm_q = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        # Only self-attention modules cache; a missing entry means "no pool allocated".
        self.attn_caches: dict[str, dict[str, torch.Tensor]] | None = {} if cross_attention_dim_head is None else None

    def _cache(self, cache_name: str) -> dict[str, torch.Tensor]:
        """Return an allocated cache pool.

        Args:
            cache_name: Name of the cache pool.

        Returns:
            The pool's tensors.

        Raises:
            RuntimeError: If no pool is allocated under ``cache_name``.
        """
        if self.attn_caches is None or cache_name not in self.attn_caches:
            msg = f"KV cache {cache_name!r} is not initialized on this attention module."
            raise RuntimeError(msg)
        return self.attn_caches[cache_name]

    def clear_pred_cache(self, cache_name: str) -> None:
        """Invalidate the slots holding *predicted* (not yet observed) keys/values."""
        if self.attn_caches is None:
            return
        cache = self.attn_caches.get(cache_name)
        if cache is None:
            return
        cache["mask"][cache["is_pred"]] = False

    def clear_cache(self, cache_name: str) -> None:
        """Drop the whole cache pool for ``cache_name``."""
        if self.attn_caches is None:
            return
        self.attn_caches.pop(cache_name, None)

    def init_kv_cache(
        self,
        cache_name: str,
        total_token_len: int,
        num_head: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> None:
        """Allocate an empty KV cache pool.

        Args:
            cache_name: Name of the cache pool.
            total_token_len: Number of slots in the pool.
            num_head: Number of attention heads.
            head_dim: Per-head dimension.
            device: Device to allocate on.
            dtype: Dtype of the cached keys/values.
            batch_size: Batch size (2 when classifier-free guidance is active).
        """
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            "k": torch.empty([batch_size, total_token_len, num_head, head_dim], device=device, dtype=dtype),
            "v": torch.empty([batch_size, total_token_len, num_head, head_dim], device=device, dtype=dtype),
            "id": torch.full((total_token_len,), -1, device=device),
            "mask": torch.zeros((total_token_len,), dtype=torch.bool, device=device),
            "is_pred": torch.zeros((total_token_len,), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name: str, key_size: int) -> torch.Tensor:
        """Reserve ``key_size`` cache slots, evicting the oldest entries if needed.

        Args:
            cache_name: Name of the cache pool.
            key_size: Number of slots required.

        Returns:
            Index tensor of the reserved slots.

        Raises:
            RuntimeError: If the pool cannot supply enough slots even after eviction.
        """
        cache = self._cache(cache_name)
        mask, ids = cache["mask"], cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)
            order = torch.argsort(ids[used])
            to_free = used[order[: key_size - free.numel()]]
            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            msg = f"KV cache exhausted: need {key_size} free slots, have {free.numel()}."
            raise RuntimeError(msg)
        return free[:key_size]

    def _next_cache_id(self, cache_name: str) -> torch.Tensor:
        """Return the monotonically increasing id to stamp on the next cache write.

        Returns:
            Scalar tensor holding the next cache generation id.
        """
        cache = self._cache(cache_name)
        ids, mask = cache["id"], cache["mask"]
        if mask.any():
            return ids[mask].max() + 1
        return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(
        self,
        cache_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_pred: bool,
    ) -> torch.Tensor:
        """Write keys/values into the cache pool.

        Args:
            cache_name: Name of the cache pool.
            key: Keys of shape ``[B, S, H, D]``.
            value: Values of shape ``[B, S, H, D]``.
            is_pred: Whether these entries come from a predicted (not observed) chunk.

        Returns:
            Index tensor of the slots that were written.
        """
        cache = self._cache(cache_name)
        slots = self.allocate_slots(cache_name, key.shape[1])
        new_id = self._next_cache_id(cache_name)
        cache["k"][:, slots] = key
        cache["v"][:, slots] = value
        cache["mask"][slots] = True
        cache["id"][slots] = new_id
        cache["is_pred"][slots] = is_pred
        return slots

    def restore_cache(self, cache_name: str, slots: torch.Tensor) -> None:
        """Release cache slots reserved by a speculative (non-committing) forward."""
        self._cache(cache_name)["mask"][slots] = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        update_cache: int = 0,
        cache_name: str = "pos",
    ) -> torch.Tensor:
        """Run attention, optionally reading from and committing to the KV cache.

        Args:
            q: Query input of shape ``[B, S_q, C]``.
            k: Key input of shape ``[B, S_kv, C]``.
            v: Value input of shape ``[B, S_kv, C]``.
            rotary_emb: Rotary factors for the query/key positions, or ``None``.
            update_cache: ``0`` = speculative (slots released afterwards), ``1`` = commit
                as predicted tokens, ``2`` = commit as observed tokens.
            cache_name: Name of the cache pool to use.

        Returns:
            Attention output of shape ``[B, S_q, C]``.
        """
        kv_cache = self.attn_caches.get(cache_name) if self.attn_caches is not None else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query).unflatten(2, (self.heads, -1))
        key = self.norm_k(key).unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2),
                )
                return torch.view_as_real(x_out * freqs).flatten(3).to(x.dtype)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        slots = None
        if kv_cache is not None:
            slots = self.update_cache(cache_name, key, value, is_pred=(update_cache == 1))
            valid = kv_cache["mask"].nonzero(as_tuple=False).squeeze(-1)
            key = kv_cache["k"][:, valid]
            value = kv_cache["v"][:, valid]

        hidden_states = self.attn_op(query, key, value)

        if update_cache == 0 and slots is not None:
            self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)
