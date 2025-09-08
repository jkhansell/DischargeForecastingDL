import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from typing import Any, Callable, Tuple

# -----------------------------
# 1. RoPE helper function
# -----------------------------
@jax.jit
def apply_rope(x):
    # x: (batch, n_heads, seq_len, head_dim)
    b, h, n, d = x.shape
    assert d % 2 == 0, "head_dim must be even for RoPE"

    half = d // 2
    freqs = jnp.arange(0, half) / half
    theta = 1.0 / (10000 ** freqs)   # (half,)
    positions = jnp.arange(n)  # (n,)

    # angles: (n, half)
    angles = positions[:, None] * theta[None, :]

    cos = jnp.cos(angles)[None, None, :, :]  # (1,1,n,half)
    sin = jnp.sin(angles)[None, None, :, :]

    # split into even/odd
    x1 = x[..., ::2]   # (b,h,n,half)
    x2 = x[..., 1::2]  # (b,h,n,half)

    x_rotated = jnp.empty_like(x)
    x_rotated = x_rotated.at[..., ::2].set(x1 * cos - x2 * sin)
    x_rotated = x_rotated.at[..., 1::2].set(x1 * sin + x2 * cos)
    return x_rotated

# -----------------------------
# 2. Rotary Multi-Head Attention
# -----------------------------
class RotarySelfAttention(nn.Module):
    d_model: int
    num_heads: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, train=True):
        # Project to q, k, v

        qkv = nn.Dense(3*self.d_model)(x)
        # qkv: (batch, Nwindow, 3*self.d_model)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # q: (batch, Nwindow, self.d_model)
        # k: (batch, Nwindow, self.d_model)
        # v: (batch, Nwindow, self.d_model)
        # Split heads
        head_dim = self.d_model // self.num_heads
        q = q.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)
        k = k.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)
        v = v.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)

        q = q.transpose(0, 2, 1, 3)  # (batch, num_heads, Nwindow, head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE
        q = apply_rope(q)
        k = apply_rope(k)

        # Attention scores
        attn_scores = jnp.einsum('...id,...jd->...ij', q, k) / jnp.sqrt(head_dim)
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_weights = nn.Dropout(self.dropout)(attn_weights, deterministic=not train)

        # Weighted sum
        out = jnp.einsum('...id,...dj->...ij', attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.d_model)
        out = nn.Dense(self.d_model)(out)
        return out

# -----------------------------
# 3. Transformer Encoder Block
# -----------------------------
class TransformerEncoderBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, train=True):
        # RoPE multi-head attention
        # x: (batch, Nwindow, self.d_model)
        attn = RotarySelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout
        )(x, train=train)
        # x: (batch, Nwindow, self.d_model)

        x = x + attn
        x = nn.LayerNorm()(x)

        # Feedforward
        mlp = nn.Sequential([
            nn.Dense(self.mlp_dim),
            nn.relu,
            nn.Dense(self.d_model),
            nn.Dropout(self.dropout, deterministic=not train)
        ])(x)

        x = x + mlp
        x = nn.LayerNorm()(x)
        return x

# -----------------------------
# 4. Transformer Encoder Stack
# -----------------------------
class TransformerEncoder(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    num_layers: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, train=True):
        for _ in range(self.num_layers):
        # x: (batch, Nwindow, self.d_model)
            x = TransformerEncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout
            )(x, train=train)
        return x

# ------------------------------
# Seq Regressor
# ------------------------------
class SeqRegressor(nn.Module):
    quantiles: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.quantiles)(x)
        return out.astype(jnp.float32)  # final output float32

class QRoPETRegressor(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    num_layers: int
    n_quantiles: int
    out_features: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, train=True):
        regressors = [SeqRegressor(quantiles=self.n_quantiles, hidden_size=self.d_model) for _ in range(self.out_features)]
        
        # Input embedding
        x = nn.Dense(self.d_model)(x)  # (batch, Nwindow, d_model)

        # Multiple CLS tokens (1 per quantile)
        cls_tokens = self.param(
            "cls_tokens",
            nn.initializers.normal(stddev=0.05),
            (1, self.out_features, self.d_model)
        )

        cls_tokens = jnp.tile(cls_tokens, (x.shape[0], 1, 1))  # (batch, nhorizons, d_model)

        # Prepend CLS tokens
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (batch, nhorizons+Nwindow, d_model)

        # Transformer encoder
        x = TransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )(x, train=train)

        # Extract CLS outputs (first n_quantiles positions)
        x_cls = x[:, :self.out_features, :]  # (batch, nhorizons, d_model)

        # Map CLS outputs to features
        #out = nn.Dense(self.out_features)(x_cls)  # (batch, n_quantiles, out_features)
        
        out = jnp.stack([regressor(x_cls[:,i,:]) for i, regressor in enumerate(regressors)], axis=1)

        return out
        
# -----------------------------
# 5. Full Time Series Transformer
# -----------------------------
"""
class QRoPETRegressor(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    num_layers: int
    n_quantiles: int
    out_features: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, train=True):
        # Create one regression head per feature
        regressors = [SeqRegressor(self.n_quantiles) for _ in range(self.out_features)]
        
        # x: (batch, Nwindow, Nfeatures)
        x = nn.Dense(self.d_model)(x)  # Input embedding
        # x: (batch, Nwindow, self.d_model)

        # Transformer encoder stack with RoPE attention
        x = TransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )(x, train=train)

        x_last = x[:,-1,:]


        # Apply each head to the embedding
        out = jnp.stack([regressor(x_last) for regressor in regressors], axis=1)
        # out shape: (batch, n_features, n_quantiles)
        return out


"""
