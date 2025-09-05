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
   # x: (Nbatch, Nwindow, d_model)
    x1 = x[..., ::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    Nbatch, Nwindow, d_model = x.shape
    head_dim = d_model // 2
    
    # rotation angles
    theta = jnp.arange(Nwindow)[:, None] / (10000 ** (jnp.arange(0, d_model, 2) / d_model))
    theta = theta[None, :, :]  # shape: (1, Nwindow, head_dim)

    # apply rotation
    x1_rot = x1 * jnp.cos(theta) - x2 * jnp.sin(theta)
    x2_rot = x1 * jnp.sin(theta) + x2 * jnp.cos(theta)
    
    # interleave
    x_rot = jnp.empty_like(x)
    x_rot = x_rot.at[..., ::2].set(x1_rot)
    x_rot = x_rot.at[..., 1::2].set(x2_rot)
    return x_rot


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

        # Apply RoPE
        q = apply_rope(q)
        k = apply_rope(k)

        # Split heads
        head_dim = self.d_model // self.num_heads
        q = q.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)
        k = k.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)
        v = v.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)

        q = q.transpose(0, 2, 1, 3)  # (batch, num_heads, Nwindow, head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

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

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.quantiles)(x)
        return out.astype(jnp.float32)  # final output float32


EPS = 1e-8

def softplus(x):
    return jnp.log1p(jnp.exp(x))

class DenseVariational(nn.Module):
    features: int
    use_bias: bool = True
    prior_std: float = 1.0

    @nn.compact
    def __call__(self, x):
        """
        x: [batch, in_dim]
        returns: [batch, features]
        Also records KL in the 'kl' mutable collection via self.sow(...)
        """
        in_dim = x.shape[-1]
        # weight posterior params
        w_mu = self.param('w_mu', nn.initializers.lecun_normal(), (in_dim, self.features))
        w_rho = self.param('w_rho', nn.initializers.normal(stddev=1e-3), (in_dim, self.features))
        w_sigma = softplus(w_rho) + EPS

        # sample weight
        key = self.make_rng('sample')
        eps_w = jax.random.normal(key, w_mu.shape)
        w = w_mu + w_sigma * eps_w

        out = jnp.dot(x, w)

        # KL for weights vs N(0, prior_std^2)
        # KL(N(mu_q, sigma_q) || N(0, sigma_p^2)) per dimension:
        # = log(sigma_p/sigma_q) + (sigma_q^2 + mu_q^2) / (2 sigma_p^2) - 1/2
        sigma_p = self.prior_std
        kl_w = jnp.sum(
            jnp.log(sigma_p / w_sigma)
            + (w_sigma**2 + w_mu**2) / (2.0 * sigma_p**2)
            - 0.5
        )

        if self.use_bias:
            b_mu = self.param('b_mu', nn.initializers.zeros, (self.features,))
            b_rho = self.param('b_rho', nn.initializers.normal(stddev=1e-3), (self.features,))
            b_sigma = softplus(b_rho) + EPS
            keyb = jax.random.normal(self.make_rng('sample'), b_mu.shape)
            b = b_mu + b_sigma * keyb
            out = out + b
            kl_b = jnp.sum(
                jnp.log(sigma_p / b_sigma)
                + (b_sigma**2 + b_mu**2) / (2.0 * sigma_p**2)
                - 0.5
            )
        else:
            kl_b = 0.0

        total_kl = kl_w + kl_b
        # record KL to be retrieved after apply(..., mutable=['kl'])
        # store under collection 'kl' with name 'kl' (could be improved by layer name)
        self.sow('ELBO', "kl", total_kl)
        return out


class BayesianMLPHead(nn.Module):
    nhorizon: int
    hidden_size: int = 128
    prior_std: float = 1.0

    @nn.compact
    def __call__(self, x):
        # Shared deterministic projection
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)

        # Separate Bayesian heads
        mu = DenseVariational(self.nhorizon, prior_std=self.prior_std)(x)
        logvar = DenseVariational(self.nhorizon, prior_std=self.prior_std)(x)

        logvar = jnp.log(EPS + softplus(logvar))  # constrain

        return mu, logvar

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
        #regressors = [SeqRegressor(self.n_quantiles) for _ in range(self.out_features)]
        regressors = [BayesianMLPHead(nhorizon=1, hidden_size=self.d_model) for _ in range(self.out_features)]

        # Input embedding
        x = nn.Dense(self.d_model)(x)  # (batch, Nwindow, d_model)

        # Multiple CLS tokens (1 per quantile)
        cls_tokens = self.param(
            "cls_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.out_features, self.d_model)
        )

        cls_tokens = jnp.tile(cls_tokens, (x.shape[0], 1, 1))  # (batch, n_quantiles, d_model)

        # Prepend CLS tokens
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (batch, n_quantiles+Nwindow, d_model)

        # Transformer encoder
        x = TransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )(x, train=train)


        x_cls = x[:, :self.out_features, :]  # (batch, n_quantiles, d_model)
    
        # Extract CLS outputs (first n_quantiles positions)
        # Map CLS outputs to features
        #out = nn.Dense(self.out_features)(x_cls)  # (batch, n_quantiles, out_features)
        #out = jnp.stack([regressor(x_cls[:,i,:]) for i,regressor in enumerate(regressors)], axis=1)

       # x_cls: transformer CLS embedding, shape (batch, d_model)
        mus, logvars = [], []

        for i, regressor in enumerate(regressors):
            mu, logvar = regressor(x_cls[:,i,:])  # each returns (batch, 1)
            mus.append(mu)
            logvars.append(logvar)

        # stack along horizon dimension
        mu = jnp.concatenate(mus, axis=1)        # (batch, nhorizon)
        logvar = jnp.concatenate(logvars, axis=1)  # (batch, nhorizon)

        return mu, logvar

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
        regressors = [SeqRegressor(self.n_quantiles) for _ in range(self.out_features)]
        
        # Input embedding
        x = nn.Dense(self.d_model)(x)  # (batch, Nwindow, d_model)

        # Multiple CLS tokens (1 per quantile)
        cls_tokens = self.param(
            "cls_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.d_model)
        )

        cls_tokens = jnp.tile(cls_tokens, (x.shape[0], 1, 1))  # (batch, n_quantiles, d_model)

        # Prepend CLS tokens
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (batch, n_quantiles+Nwindow, d_model)

        # Transformer encoder
        x = TransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )(x, train=train)

        # Extract CLS outputs (first n_quantiles positions)
        x_cls = x[:, 0, :]  # (batch, n_quantiles, d_model)

        # Map CLS outputs to features
        #out = nn.Dense(self.out_features)(x_cls)  # (batch, n_quantiles, out_features)
        
        out = jnp.stack([regressor(x_cls) for i,regressor in enumerate(regressors)], axis=1)

        return out
"""
