from __future__ import annotations

"""ST-BERT style model used in Plan C.

This file is intentionally self-contained and defensive.

Bugfix note (important):
------------------------
We must NOT keep direct Python-list references to registered buffers.
When you call `model.to(device)`, PyTorch replaces buffers in-place.
If we store the old CPU tensor in a list (e.g. `self.token2regions.append(buf)`),
that list will still point to the old CPU tensor after `.to(cuda)`.
Indexing with CUDA indices then crashes:

  RuntimeError: indices should be either on cpu or on the same device...

Therefore we only store buffer *names* and retrieve them via getattr() inside
forward; this always returns the current buffer on the correct device.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn


def _make_causal_mask(L: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular causal mask for TransformerEncoder.

    In torch TransformerEncoder, `mask=True` positions are NOT allowed.
    """
    return torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)


class Fourier2D(nn.Module):
    def __init__(self, m: int = 8):
        super().__init__()
        self.m = int(m)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """2D Fourier features.

        Args:
            xy: [..., 2] normalized coordinates
        Returns:
            [..., 4*m]
        """
        m = self.m
        freqs = (2.0 ** torch.arange(m, device=xy.device, dtype=xy.dtype)) * np.pi
        x = xy[..., 0:1]
        y = xy[..., 1:2]
        xw = x * freqs
        yw = y * freqs
        return torch.cat([torch.sin(xw), torch.cos(xw), torch.sin(yw), torch.cos(yw)], dim=-1)


@dataclass
class STBertConfig:
    vocab_size: int
    num_users: int
    num_cats: int
    num_tod: int  # includes 0=unknown
    num_dow: int  # includes 0=unknown
    region_vocab_sizes: List[int]  # each includes 0=unknown
    max_len: int = 128
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    user_dim: int = 128
    pos2d_freq: int = 8
    tie_weights: bool = False


class STBert(nn.Module):
    """Spatio-Temporal BERT for POI trajectories.

    Supports:
      - MLM pretraining (bidirectional)
      - Next-POI fine-tuning (causal)
    """

    def __init__(
        self,
        cfg: STBertConfig,
        *,
        token2cat: np.ndarray,
        token2regions: List[np.ndarray],
        token2xy: np.ndarray,
        pad_token: int = 0,
        mask_token: int = 1,
        cls_token: int = 2,
    ):
        super().__init__()
        self.cfg = cfg
        self.pad_token = int(pad_token)
        self.mask_token = int(mask_token)
        self.cls_token = int(cls_token)

        # ---- buffers for token->meta (shape [V] or [V,2]) ----
        # NOTE: they must be registered buffers so they move with `.to(device)`.
        self.register_buffer("token2cat", torch.as_tensor(token2cat, dtype=torch.long), persistent=True)
        self.register_buffer("token2xy", torch.as_tensor(token2xy, dtype=torch.float32), persistent=True)

        self.num_region_scales = len(token2regions)
        self._token2region_names: List[str] = []
        for i, t2r in enumerate(token2regions):
            name = f"token2region_{i}"
            self.register_buffer(name, torch.as_tensor(t2r, dtype=torch.long), persistent=True)
            self._token2region_names.append(name)

        # ---- embeddings ----
        self.poi_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        self.tod_emb = nn.Embedding(cfg.num_tod, cfg.d_model)  # 0 unknown
        self.dow_emb = nn.Embedding(cfg.num_dow, cfg.d_model)  # 0 unknown

        self.cat_emb = nn.Embedding(cfg.num_cats, cfg.d_model)  # 0 unknown
        self.region_embs = nn.ModuleList([nn.Embedding(vsz, cfg.d_model) for vsz in cfg.region_vocab_sizes])
        assert len(self.region_embs) == self.num_region_scales, "region_vocab_sizes and token2regions mismatch"

        self.user_emb = nn.Embedding(cfg.num_users, cfg.user_dim)
        self.user_proj = nn.Linear(cfg.user_dim, cfg.d_model)

        # 2D Fourier encoding
        self.fourier2d = Fourier2D(m=cfg.pos2d_freq)
        self.pos2d_proj = nn.Sequential(
            nn.Linear(4 * cfg.pos2d_freq, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.emb_ln = nn.LayerNorm(cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        # ---- heads ----
        self.mlm_poi = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.mlm_cat = nn.Linear(cfg.d_model, cfg.num_cats)
        self.mlm_regions = nn.ModuleList([nn.Linear(cfg.d_model, vsz) for vsz in cfg.region_vocab_sizes])

        self.next_poi = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.next_cat = nn.Linear(cfg.d_model, cfg.num_cats)
        self.next_regions = nn.ModuleList([nn.Linear(cfg.d_model, vsz) for vsz in cfg.region_vocab_sizes])

        if cfg.tie_weights:
            # tie POI embedding with MLM head
            self.mlm_poi.weight = self.poi_emb.weight

    def _get_token2region(self, scale_idx: int) -> torch.Tensor:
        """Get token->region buffer for a scale.

        Must use getattr() to avoid stale CPU references after `.to(cuda)`.
        """
        return getattr(self, self._token2region_names[scale_idx])

    def _embed_tokens(
        self,
        input_tokens: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        user: torch.Tensor,
    ) -> torch.Tensor:
        """Token embedding for both MLM and next-POI.

        Args:
            input_tokens: [B,L]
            tod/dow: [B,L], already shifted (0 unknown)
            user: [B]
        Returns:
            x: [B,L,D]
        """
        if input_tokens.dim() != 2:
            raise ValueError(f"input_tokens must be [B,L], got {tuple(input_tokens.shape)}")
        B, L = input_tokens.shape
        device = input_tokens.device

        tok_e = self.poi_emb(input_tokens)  # [B,L,D]
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_e = self.pos_emb(pos_ids)

        tod_e = self.tod_emb(tod)
        dow_e = self.dow_emb(dow)

        # token meta (buffer indexing must be on same device)
        # token2cat/token2xy are registered buffers, so they move with the model.
        cat_id = self.token2cat[input_tokens]  # [B,L]
        cat_e = self.cat_emb(cat_id)

        reg_sum = torch.zeros_like(tok_e)
        for s, emb in enumerate(self.region_embs):
            t2r = self._get_token2region(s)
            rid = t2r[input_tokens]
            reg_sum = reg_sum + emb(rid)

        xy = self.token2xy[input_tokens]  # [B,L,2]
        pos2d = self.pos2d_proj(self.fourier2d(xy))

        u = self.user_proj(self.user_emb(user))  # [B,D]
        u = u.unsqueeze(1).expand(-1, L, -1)

        x = tok_e + pos_e + tod_e + dow_e + cat_e + reg_sum + pos2d + u
        x = self.emb_ln(x)
        x = self.emb_drop(x)
        return x

    def encode(
        self,
        input_tokens: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        attn_mask: torch.Tensor,
        user: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        """Encode sequences.

        Returns hidden states [B,L,D].
        """
        # Ensure boolean
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()

        x = self._embed_tokens(input_tokens, tod, dow, user)
        key_padding_mask = ~attn_mask  # True for PAD
        if causal:
            L = input_tokens.size(1)
            attn_mask2 = _make_causal_mask(L, device=input_tokens.device)
        else:
            attn_mask2 = None
        h = self.encoder(x, mask=attn_mask2, src_key_padding_mask=key_padding_mask)
        return h

    def forward_mlm(
        self,
        input_tokens: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        attn_mask: torch.Tensor,
        user: torch.Tensor,
    ):
        """MLM forward (bidirectional)."""
        h = self.encode(input_tokens, tod, dow, attn_mask, user, causal=False)
        poi_logits = self.mlm_poi(h)
        cat_logits = self.mlm_cat(h)
        reg_logits = [head(h) for head in self.mlm_regions]  # list of [B,L,R]
        return poi_logits, cat_logits, reg_logits

    def forward_next(
        self,
        input_tokens: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        attn_mask: torch.Tensor,
        user: torch.Tensor,
        query_tod: torch.Tensor,
        query_dow: torch.Tensor,
        *,
        causal: bool = True,
    ):
        """Next-POI forward.

        Returns logits over vocab (including special tokens; caller should mask invalid ones).
        """
        h = self.encode(input_tokens, tod, dow, attn_mask, user, causal=causal)  # [B,L,D]
        B, L, D = h.shape

        # pick last valid position (includes CLS at pos0)
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()
        lengths = attn_mask.long().sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(B, 1, 1).expand(-1, 1, D)
        h_last = h.gather(1, idx).squeeze(1)  # [B,D]

        # Add query time embedding + user embedding
        q = h_last + self.tod_emb(query_tod) + self.dow_emb(query_dow) + self.user_proj(self.user_emb(user))
        q = torch.tanh(q)

        poi_logits = self.next_poi(q)
        cat_logits = self.next_cat(q)
        reg_logits = [head(q) for head in self.next_regions]
        return poi_logits, cat_logits, reg_logits
