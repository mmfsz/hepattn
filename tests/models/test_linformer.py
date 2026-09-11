import pytest
import torch

from hepattn.models.attention import Attention
from hepattn.models.linformer import LinformerAttention

torch.manual_seed(42)


@pytest.mark.parametrize("kv_len", [None, 96, 160])
def test_linformer_shapes_and_padding(kv_len):
    batch_size, q_len, dim, num_heads = 3, 150, 64, 8
    attn = Attention(dim=dim, num_heads=num_heads, attn_type="linformer", linformer_seq_len=160, linformer_proj_dim=16)
    assert isinstance(attn.attn, LinformerAttention)
    assert not hasattr(attn, "in_proj_weight")

    q = torch.randn(batch_size, q_len, dim)
    kv = torch.randn(batch_size, kv_len, dim) if kv_len else None
    kv_mask = torch.rand(batch_size, kv_len or q_len) > 0.3
    kv_mask[:, 0] = True

    out = attn(q, kv, kv_mask=kv_mask)
    assert out.shape == (batch_size, q_len, dim)
    assert torch.isfinite(out).all()

    # Padded key/value slots must not influence the output: replace them with garbage and compare
    if kv is not None:
        kv_garbage = torch.where(kv_mask.unsqueeze(-1), kv, 1e3 * torch.randn_like(kv))
        out_garbage = attn(q, kv_garbage, kv_mask=kv_mask)
        torch.testing.assert_close(out, out_garbage)


def test_linformer_rejects_attention_mask_and_backend_switch():
    attn = Attention(dim=32, num_heads=4, attn_type="linformer", linformer_seq_len=8, linformer_proj_dim=4)
    q = torch.randn(2, 8, 32)
    with pytest.raises(AssertionError):
        attn(q, attn_mask=torch.ones(2, 8, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="linformer"):
        attn.set_backend("torch")
    with pytest.raises(ValueError, match="linformer"):
        Attention(dim=32, num_heads=4, attn_type="torch").set_backend("linformer")


def test_linformer_set_backend_keeps_weights():
    attn = Attention(dim=32, num_heads=4, attn_type="linformer", linformer_seq_len=8, linformer_proj_dim=4)
    proj = attn.attn.proj_k.detach().clone()
    attn.set_backend("linformer")
    torch.testing.assert_close(attn.attn.proj_k, proj)


def test_linformer_sequence_too_long():
    attn = Attention(dim=32, num_heads=4, attn_type="linformer", linformer_seq_len=8, linformer_proj_dim=4)
    with pytest.raises(AssertionError, match="at most 8"):
        attn(torch.randn(1, 9, 32))
