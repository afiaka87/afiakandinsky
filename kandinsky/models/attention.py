import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from ..device_utils import compile_if_cuda, is_mps_device

try:
    from flash_attn import flash_attn_func as flash_attention_2
    print("FlashAttention 2 is found")
except:
    flash_attention_2 = None

try:
    from flash_attn_interface import flash_attn_func as flash_attention_3
    print("FlashAttention 3 is found")
except:
    flash_attention_3 = None

try:
    import sageattention
    print(f"Sage Attention is found")
except:
    sageattention = None


def chunked_sdpa_mps(q, k, v, chunk_size=256):
    """
    Memory-efficient chunked attention for MPS.
    Processes attention in smaller chunks to avoid OOM on MPS backend.

    Args:
        q: Query tensor [B, N, H, D]
        k: Key tensor [B, N, H, D]
        v: Value tensor [B, N, H, D]
        chunk_size: Number of query tokens to process at once

    Returns:
        Output tensor [B, N, H, D]
    """
    B, N, H, D = q.shape
    scale = D ** -0.5

    # Transpose to [B, H, N, D] for attention computation
    q = q.transpose(1, 2)  # [B, H, N, D]
    k = k.transpose(1, 2)  # [B, H, N, D]
    v = v.transpose(1, 2)  # [B, H, N, D]

    # Compute attention in chunks along query dimension
    output_chunks = []
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        q_chunk = q[:, :, i:end_i, :]  # [B, H, chunk, D]

        # Compute attention scores for this chunk
        attn_weights = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale  # [B, H, chunk, N]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        out_chunk = torch.matmul(attn_weights, v)  # [B, H, chunk, D]
        output_chunks.append(out_chunk)

    # Concatenate chunks
    out = torch.cat(output_chunks, dim=2)  # [B, H, N, D]
    out = out.transpose(1, 2).contiguous()  # [B, N, H, D]

    return out


@compile_if_cuda(mode="max-autotune-no-cudagraphs", dynamic=True)
def sdpa(q, k, v):
    # Check if we're on MPS - use chunked attention
    if q.device.type == "mps":
        return chunked_sdpa_mps(q, k, v, chunk_size=256)

    # CUDA path - use native SDPA
    query = q.transpose(1, 2).contiguous()
    key = k.transpose(1, 2).contiguous()
    value = v.transpose(1, 2).contiguous()
    out = (
        F.scaled_dot_product_attention(
            query,
            key,
            value
        )
        .transpose(1, 2)
        .contiguous()
    )
    return out

@compile_if_cuda(mode="max-autotune-no-cudagraphs", dynamic=True)
def sage_attn(q, k, v):
    out = (
        sageattention.sageattn(
            q, k, v,
            tensor_layout="NHD",
            is_causal=False
        )
    )
    return out

class SelfAttentionEngine():
    def __init__(self, engine="auto"):
        assert engine in ["auto", "flash_attention_2", "flash_attention_3", "sage", "sdpa"]
        self.attention_fn = None

        if engine == "flash_attention_2":
            if flash_attention_2 is None:
                raise RuntimeError("flash_attention_2 engine selected, but it can't be imported.")
            self.attention_fn = flash_attention_2

        if engine == "flash_attention_3":
            if flash_attention_3 is None:
                raise RuntimeError("flash_attention_3 engine selected, but it can't be imported.")
            self.attention_fn = flash_attention_3

        if engine == "sage":
            if sageattention is None:
                raise RuntimeError("sage engine selected, but it can't be imported.")
            self.attention_fn = sage_attn

        if engine == "sdpa":
            self.attention_fn = sdpa
        
        if engine == "auto":
            self.attention_fn = sdpa
            if not sageattention is None:
                self.attention_fn = sage_attn
            if not flash_attention_2 is None:
                self.attention_fn = flash_attention_2
            if not flash_attention_3 is None:
                self.attention_fn = flash_attention_3
    
    def get_attention(self):
        return self.attention_fn

