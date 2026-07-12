import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        output = torch.cat((o1, o2), dim=-1)
    else:
        output = torch.stack((o1, o2), dim=-1).flatten(-2)
    return output

def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = apply_rotary_emb(
        query_rot,
        cos,
        sin,
        is_neox_style,
    )
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, head_size)
        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]
        key_rot = apply_rotary_emb(
            key_rot,
            cos,
            sin,
            is_neox_style,
        )
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


class RotaryEmbedding(RotaryEmbeddingBase):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
    ) -> None:
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
            init_cache=init_cache,
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A PyTorch-native implementation of forward()."""
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        return rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox_style,
        )