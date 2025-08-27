import jax 
import jax.numpy as jnp

import flax.linen as nn
from flax.training import train_state
from typing import Any, Callable, Tuple

def causal_mask(T: int):
    # (1, 1, T, T) broadcastable mask: True = keep, False = mask
    m = jnp.tril(jnp.ones((T, T), dtype=bool))
    return m[None, None, :, :]

class MLPBlock(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = nn.Dense(4 * self.hidden_size)(y)
        y = nn.relu(y)
        y = nn.Dense(self.hidden_size)(y)
        return x + y

class QATNEncoderBlock(nn.Module):
    hidden_size: int
    n_heads: int
    causal: bool = True
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x_tf, feat_tokens, train: bool): 
        B, T, d = x_tf.shape
        F = feat_tokens.shape[1]

        h = nn.LayerNorm()(x_tf)
        mask = causal_mask(T) if self.causal else None
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            deterministic = not train
        )(h, h, mask=mask)
    
        x_tf = x_tf + h


        ft = nn.LayerNorm()(feat_tokens)
        keys_vals = nn.LayerNorm()(x_tf)
        ft_upd = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
        )(ft, keys_vals, mask=None)
        feat_tokens = feat_tokens + ft_upd

        # --- Time queries features (return flow) ---
        h2 = nn.LayerNorm()(x_tf)
        kv_feat = nn.LayerNorm()(feat_tokens)
        t_upd = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
        )(h2, kv_feat, mask=None)
        x_tf = x_tf + t_upd

        # --- MLPs ---
        x_tf = MLPBlock(self.hidden_size)(x_tf)
        feat_tokens = MLPBlock(self.hidden_size)(feat_tokens)
         
        return x_tf, feat_tokens

class QATNEncoder(nn.Module):
    hidden_size: int
    depth: int = 4
    n_heads: int = 8
    causal: bool = True
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        x: (B, T, F)  -- your input; if you have (T,F), add a batch dim B=1
        returns: temporal embeddings (B,T,d), feature embeddings (B,F,d)
        """
        B, T, F = x.shape

        # Project features into model dim per time step
        # (treat features as channels; Dense mixes features at each time)
        h_t = nn.Dense(self.hidden_size)(x)  # (B, T, d)

        # Add learned temporal positions
        pos_t = self.param("pos_time", nn.initializers.normal(0.02), (T, self.hidden_size))
        h_t = h_t + pos_t[None, :, :]

        # Initialize learned feature tokens (shared across batch)
        feat_emb = self.param("feat_tokens", nn.initializers.normal(0.02), (F, self.hidden_size))
        feat_tokens = jnp.broadcast_to(feat_emb[None, :, :], (B, F, self.hidden_size))

        # (Optional) add feature index encodings if F changes across datasets
        # pos_f = self.param("pos_feat", nn.initializers.normal(0.02), (F, self.hidden_size))
        # feat_tokens = feat_tokens + pos_f[None, :, :]

        for _ in range(self.depth):
            h_t, feat_tokens = QATNEncoderBlock(
                hidden_size=self.hidden_size,
                n_heads=self.n_heads,
                causal=self.causal,
                dropout=self.dropout,
            )(h_t, feat_tokens, train=train)

        return h_t, feat_tokens  # temporal & feature representations

class SeqRegressor(nn.Module):
    features: int
    quantiles: int

    @nn.compact
    def __call__(self, x):
        """
        x: (batch, features)
        returns: (batch, features, quantiles)
        """
        out = nn.Dense(self.features * self.quantiles)(x)
        return out

class QATNRegressor(nn.Module):
    features: int
    quantiles: int
    hidden_size: int
    depth: int
    n_heads: int
    causal: bool
    dropout: float

    @nn.compact
    def __call__(self, x):
        qatn = QATNEncoder(
            hidden_size=self.hidden_size, 
            depth=self.depth, 
            n_heads=self.n_heads, 
            causal=self.causal, 
            dropout=self.dropout
        )
        
        regressors = [SeqRegressor(1, self.quantiles) for _ in range(self.features)]

        x_t, x_f = qatn(x)
        # Pool time once (global context)
        t_repr = jnp.mean(x_t, axis=1)              # (B,d)

        # Optionally also a pooled feature context (helps small F)
        f_global = jnp.mean(x_f, axis=1)            # (B,d)

        # Build per-feature inputs: [feature_token || time_pool || feat_pool]
        B, F, d = x_f.shape
        g = jnp.concatenate([t_repr, f_global], axis=-1)          # (B, 2d)
        g = jnp.broadcast_to(g[:, None, :], (B, F, 2*d))          # (B,F,2d)
        per_feat = jnp.concatenate([x_f, g], axis=-1)             # (B,F,3d)
    
        out = jnp.stack([regressor(x[:,i,:]) for i, regressor in enumerate(regressors)],axis=1)

        return out


class QATNTrainState(train_state.TrainState):
    pass

@jax.jit
def QATNtrain_step(state, batch, quantiles):
    """
    state: TrainState
    batch: dict with "x" and "y"
        x: (batch, time, input_features)
        y: (batch, features)
    """
    def loss_fn(params):
        preds = state.apply_fn(params, batch['x'])
        #loss = quantile_loss_complex(
        #    preds, batch['y'], quantiles,
        #    crossing_penalty_coef=0.1, mae_coef=0.5
        #)

        loss = quantile_loss(preds, batch["y"], quantiles)
        return loss 

    # Get both loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.tree_util.tree_map(
        lambda g: jax.lax.pmean(g, axis_name="batch"), grads
    )

    # Update state
    state = state.apply_gradients(grads=grads)
    loss = jax.tree_util.tree_map(
        lambda x: jax.lax.pmean(x, axis_name="batch"), loss
    )
    
    return state, loss

@jax.jit
def QATNeval_step(state, batch, quantiles):
    """
    state: TrainState
    batch: dict with "x" and "y"
        x: (batch, time, input_features)
        y: (batch, features)
    """
    def loss_fn(params):
        preds = state.apply_fn(params, batch['x'])
        loss = quantile_loss(preds, batch["y"], quantiles)

        return loss, preds
    
    loss, preds = loss_fn(state.params)
    loss = jax.tree_util.tree_map(
        lambda x: jax.lax.pmean(x, axis_name="batch"), loss
    )
    return loss, preds
