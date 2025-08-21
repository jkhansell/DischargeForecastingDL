import jax
import jax.numpy as jnp

from flax.linen import nn
from typing import Any, Callable, Tuple

class LTCNCell(nn.Module):
    hidden_size: int
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, carry, inputs, dt):
        h = carry  # shape (hidden_size,)
        input_dim = inputs.shape[-1]

        gamma = self.param("gamma", nn.initializers.xavier_uniform(),
                           (input_dim, self.hidden_size))
        gamma_r = self.param("gamma_r", nn.initializers.xavier_uniform(),
                             (self.hidden_size, self.hidden_size))
        mu = self.param("mu", nn.initializers.uniform(),
                        (self.hidden_size,))
        A = self.param("A", nn.initializers.uniform(),
                       (self.hidden_size,))
        tau = self.param("tau", nn.initializers.constant(1.0),
                         (self.hidden_size,))

        # --- Encode input ---
        I_t = jnp.dot(inputs, gamma)            # (hidden_size,)
        x_t = jnp.dot(h, gamma_r) + mu              # (hidden_size,)

        # --- Nonlinearity ---
        f = self.activation(x_t + I_t)

        # --- Liquid time-constant integration ---
        x_tp1 = (x_t + dt * f * A) / (1 + dt * (1 / tau + f))

        return x_tp1, x_tp1

class LTCN(nn.Module):
    hidden_size: int
    
    @nn.compact
    def __call__(self, x):
        B, T, D = x.shape
        h0 = jnp.zeros((B, self.hidden_size))

        ltcn_scan = nn.scan(
            LTCNCell, 
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,    # time axis in input
            out_axes=1    # keep time axis in outputs
        )(self.hidden_size, name="ltcn_cell")
    
        x_final, outputs = ltcn_scan(h0, x)
        return outputs

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
        return out.reshape(x.shape[0], self.features, self.quantiles)

class LTCNRegressor(nn.Module):
    features: int
    quantiles: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        lstm = LTCN(hidden_size=self.hidden_size)
        regressor = SeqRegressor(self.features, self.quantiles)

        x = lstm(x)
        out = regressor(x[:,-1,:])

        return out

class LTCNTrainState(train_state.TrainState):
    pass


@jax.jit
def quantile_loss_complex(
    y_pred,
    y_true,
    quantiles,
    crossing_penalty_coef=0.0,
    mae_coef=0.0
):
    """
    Flexible quantile (pinball) loss using JAX-lax conditionals for JIT stability.
    """

    # Ensure y_true is (Nstations, 1)

    # Error across all quantiles
    error = y_true - y_pred  # (Nstations, Nquantiles)
    mae = jnp.mean(jnp.abs(y_true - y_pred[..., y_pred.shape[-1]//2, None]))

    # Pinball loss
    loss = jnp.maximum(quantiles * error, (quantiles - 1) * error)  # (Nstations, Nquantiles)
    loss = jnp.mean(loss)
    # Crossing penalty
    def compute_penalty(_):
        return jnp.mean(jnp.maximum(0, y_pred[:, :-1] - y_pred[:, 1:]))
    crossing_penalty = jax.lax.cond(crossing_penalty_coef > 0.0, compute_penalty, lambda _: 0.0, operand=None)

    # Weighted loss

    # Combine
    total_loss = loss + crossing_penalty_coef * crossing_penalty + mae_coef * mae

    # Return either per-quantile or total
    return total_loss


@jax.jit
def LTCNtrain_step(state, batch, quantiles):
    """
    state: TrainState
    batch: dict with "x" and "y"
        x: (batch, time, input_features)
        y: (batch, features)
    """
    def loss_fn(params):
        preds = state.apply_fn(params, batch['x'])
        loss = quantile_loss_complex(
            preds, batch['y'], quantiles,
            crossing_penalty_coef=0.5, mae_coef=0.5
        )
        
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
def LTCNeval_step(state, batch, quantiles):
    """
    state: TrainState
    batch: dict with "x" and "y"
        x: (batch, time, input_features)
        y: (batch, features)
    """
    def loss_fn(params):
        preds = state.apply_fn(params, batch['x'])
        loss = quantile_loss_complex(
            preds, batch['y'], quantiles,
            crossing_penalty_coef=0.5, mae_coef=0.5
        )
        return loss, preds
    
    loss, preds = loss_fn(state.params)
    loss = jax.tree_util.tree_map(
        lambda x: jax.lax.pmean(x, axis_name="batch"), loss
    )
    return loss, preds
    