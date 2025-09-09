import jax 
import jax.numpy as jnp

import flax.linen as nn
from typing import Any, Callable, Tuple

# ------------------------------
# Utilities
# ------------------------------
def causal_mask(T: int, compute_dtype=jnp.bfloat16):
    # (1, 1, T, T) broadcastable mask: True = keep, False = mask
    m = jnp.tril(jnp.ones((T, T), dtype=bool))
    return m[None, None, :, :]

# ------------------------------
# MLP Block
# ------------------------------
class MLPBlock(nn.Module):
    hidden_size: int
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.compute_dtype)
        y = nn.LayerNorm(dtype=self.compute_dtype)(x)
        y = nn.Dense(4 * self.hidden_size, param_dtype=jnp.float32, dtype=self.compute_dtype)(y)
        y = nn.relu(y)
        y = nn.Dense(self.hidden_size, param_dtype=jnp.float32, dtype=self.compute_dtype)(y)
        return x + y

# ------------------------------
# QATN Encoder Block
# ------------------------------
class QATNEncoderBlock(nn.Module):
    hidden_size: int
    n_heads: int
    causal: bool = True
    dropout: float = 0.0
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x_tf, feat_tokens, train: bool):
        x_tf = x_tf.astype(self.compute_dtype)
        feat_tokens = feat_tokens.astype(self.compute_dtype)

        B, T, d = x_tf.shape

        h = nn.LayerNorm(dtype=self.compute_dtype)(x_tf)
        mask = causal_mask(T, self.compute_dtype) if self.causal else None
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32
        )(h, h, mask=mask)
        x_tf = x_tf + h

        ft = nn.LayerNorm(dtype=self.compute_dtype)(feat_tokens)
        keys_vals = nn.LayerNorm(dtype=self.compute_dtype)(x_tf)
        ft_upd = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32
        )(ft, keys_vals, mask=None)
        feat_tokens = feat_tokens + ft_upd

        h2 = nn.LayerNorm(dtype=self.compute_dtype)(x_tf)
        kv_feat = nn.LayerNorm(dtype=self.compute_dtype)(feat_tokens)
        t_upd = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32
        )(h2, kv_feat, mask=None)
        x_tf = x_tf + t_upd

        x_tf = MLPBlock(self.hidden_size, self.compute_dtype)(x_tf)
        feat_tokens = MLPBlock(self.hidden_size, self.compute_dtype)(feat_tokens)

        return x_tf, feat_tokens

# ------------------------------
# QATN Encoder
# ------------------------------
class QATNEncoder(nn.Module):
    hidden_size: int
    depth: int = 4
    n_heads: int = 8
    causal: bool = True
    dropout: float = 0.0
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, train: bool = True):
        B, T, F = x.shape
        x = x.astype(self.compute_dtype)

        h_t = nn.Dense(self.hidden_size, param_dtype=jnp.float32, dtype=self.compute_dtype)(x)

        pos_t = self.param("pos_time", nn.initializers.normal(0.02), (T, self.hidden_size))
        h_t = h_t + pos_t[None, :, :].astype(self.compute_dtype)

        feat_emb = self.param("feat_tokens", nn.initializers.normal(0.02), (F, self.hidden_size))
        feat_tokens = jnp.broadcast_to(feat_emb[None, :, :], (B, F, self.hidden_size)).astype(self.compute_dtype)

        for _ in range(self.depth):
            h_t, feat_tokens = QATNEncoderBlock(
                hidden_size=self.hidden_size,
                n_heads=self.n_heads,
                causal=self.causal,
                dropout=self.dropout,
                compute_dtype=self.compute_dtype
            )(h_t, feat_tokens, train=train)

        return h_t, feat_tokens

# ------------------------------
# Seq Regressor
# ------------------------------
class SeqRegressor(nn.Module):
    features: int
    quantiles: int
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.compute_dtype)
        out = nn.Dense(self.features * self.quantiles, param_dtype=jnp.float32, dtype=self.compute_dtype)(x)
        return out.astype(jnp.float32)  # final output float32

# ------------------------------
# QATN Regressor
# ------------------------------
class QATNRegressor(nn.Module):
    features: int
    quantiles: int
    hidden_size: int
    depth: int
    n_heads: int
    causal: bool = False
    dropout: float = 0.1
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, train=True):
        x = x.astype(self.compute_dtype)
        qatn = QATNEncoder(
            hidden_size=self.hidden_size,
            depth=self.depth,
            n_heads=self.n_heads,
            causal=self.causal,
            dropout=self.dropout,
            compute_dtype=self.compute_dtype
        )

        regressors = [SeqRegressor(1, self.quantiles, self.compute_dtype) for _ in range(self.features)]

        x_t, x_f = qatn(x, train)
        t_repr = jnp.mean(x_t, axis=1).astype(self.compute_dtype)
        f_global = jnp.mean(x_f, axis=1).astype(self.compute_dtype)

        B, F, d = x_f.shape
        g = jnp.concatenate([t_repr, f_global], axis=-1)
        g = jnp.broadcast_to(g[:, None, :], (B, F, 2*d))
        per_feat = jnp.concatenate([x_f, g], axis=-1)

        # Per-feature regression
        out = jnp.stack([regressor(per_feat[:, i, :]) for i, regressor in enumerate(regressors)], axis=1)
        return out  # float32

