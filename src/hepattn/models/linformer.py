"""Linformer attention: linear-complexity attention via a low-rank projection of the keys and values.

A trivial rewriting of https://github.com/lucidrains/linformer (arXiv:2006.04768) into a form that
fits the hepattn attention/decoder setup. Ported from lgray/compression-work (76f2078).

The keys and values of a sequence of length ``n`` are projected along the sequence axis down to
``k`` "virtual" tokens with a learned ``(seq_len, k)`` matrix, so the attention matrix is ``(n, k)``
instead of ``(n, n)``. The projection mixes sequence positions, which is why per-position attention
masks cannot be honoured: padded keys/values are zeroed before the projection instead, and an
explicit attention mask is refused.
"""

import math

import torch
from torch import Tensor, nn


def _init_projection(tensor: Tensor) -> Tensor:
    std = 1 / math.sqrt(tensor.shape[-1])
    tensor.uniform_(-std, std)
    return tensor


class LinformerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        k: int = 256,
        heads: int = 8,
        dim_head: int | None = None,
        one_kv_head: bool = False,
        share_kv: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Linformer attention module with its own input and output projections.

        Parameters
        ----------
        dim : int
            Embedding dimension of the inputs and the output.
        seq_len : int
            Maximum key/value sequence length. Shorter sequences use a slice of the projection.
        k : int
            Projected sequence length (the low rank). Values at or above ``seq_len`` compress nothing.
        heads : int
            Number of attention heads.
        dim_head : int | None
            Dimension per head; defaults to ``dim // heads``.
        one_kv_head : bool
            Share a single key/value head across all query heads.
        share_kv : bool
            Use the key projection for the values too.
        dropout : float
            Dropout on the attention weights.
        """
        super().__init__()
        assert dim % heads == 0, "dimension must be divisible by the number of heads"

        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        self.dim_head = dim_head if dim_head is not None else dim // heads
        self.share_kv = share_kv

        self.to_q = nn.Linear(dim, self.dim_head * heads, bias=False)

        kv_dim = self.dim_head if one_kv_head else self.dim_head * heads
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(_init_projection(torch.zeros(seq_len, k)))

        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(_init_projection(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(self.dim_head * heads, dim)

    def forward(self, q: Tensor, kv: Tensor | None = None, kv_mask: Tensor | None = None) -> Tensor:
        """Attend from ``q`` to ``kv`` (or to ``q`` itself for self-attention).

        Parameters
        ----------
        q : Tensor
            Queries of shape (B, N, D).
        kv : Tensor | None
            Keys/values of shape (B, M, D); ``q`` if None.
        kv_mask : Tensor | None
            Boolean (B, M) mask, True for valid key/value slots. Padded slots are zeroed before the
            sequence projection so they contribute nothing to the projected keys and values.
        """
        if kv is None:
            kv = q
        batch_size, num_queries, _ = q.shape
        kv_len = kv.shape[1]
        assert kv_len <= self.seq_len, f"the key/value sequence length must be at most {self.seq_len}, got {kv_len}"

        queries = self.to_q(q)
        keys = self.to_k(kv)
        values = self.to_v(kv) if not self.share_kv else keys

        if kv_mask is not None:
            valid = kv_mask.unsqueeze(-1).to(keys.dtype)
            keys = keys * valid
            values = values * valid

        # Allow variable sequence lengths (up to the maximum) by slicing the projections
        proj_k = self.proj_k[:kv_len]
        proj_v = self.proj_v[:kv_len] if not self.share_kv else proj_k

        # Project keys and values along the sequence axis down to k virtual tokens
        keys = torch.einsum("bnd,nk->bkd", keys, proj_k)
        values = torch.einsum("bnd,nk->bkd", values, proj_v)

        # Split heads: queries (B, H, N, Dh); keys/values (B, H, k, Dh), broadcast if a single kv head
        queries = queries.reshape(batch_size, num_queries, self.heads, self.dim_head).transpose(1, 2)
        keys = keys.reshape(batch_size, self.k, -1, self.dim_head).transpose(1, 2).expand(-1, self.heads, -1, -1)
        values = values.reshape(batch_size, self.k, -1, self.dim_head).transpose(1, 2).expand(-1, self.heads, -1, -1)

        scores = torch.einsum("bhnd,bhkd->bhnk", queries, keys) * (self.dim_head**-0.5)
        attn = self.dropout(scores.softmax(dim=-1))
        out = torch.einsum("bhnk,bhkd->bhnd", attn, values)

        out = out.transpose(1, 2).reshape(batch_size, num_queries, -1)
        return self.to_out(out)
