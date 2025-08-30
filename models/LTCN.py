import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

from typing import Any, Callable, Tuple

class LTCNCell(nn.Module):
    hidden_size: int
    dt: float
    activation: Callable = nn.tanh
    train: bool = True
    compute_dtype: jnp.dtype = jnp.bfloat16  

    @nn.compact
    def __call__(self, carry, inputs):
        h = carry.astype(self.compute_dtype)
        inputs = inputs.astype(self.compute_dtype)
        input_dim = inputs.shape[-1]

        gamma = self.param("gamma", nn.initializers.xavier_uniform(), (input_dim, self.hidden_size))
        gamma_r = self.param("gamma_r", nn.initializers.xavier_uniform(), (self.hidden_size, self.hidden_size))
        mu = self.param("mu", nn.initializers.uniform(), (self.hidden_size,))
        A = self.param("A", nn.initializers.uniform(), (self.hidden_size,))
        tau = self.param("tau", nn.initializers.constant(1.0), (self.hidden_size,))

        gamma_bf16 = gamma.astype(self.compute_dtype)
        gamma_r_bf16 = gamma_r.astype(self.compute_dtype)
        mu_bf16 = mu.astype(self.compute_dtype)
        A_bf16 = A.astype(self.compute_dtype)
        tau_bf16 = tau.astype(self.compute_dtype)

        I_t = jnp.dot(inputs, gamma_bf16)
        x_t = jnp.dot(h, gamma_r_bf16) + mu_bf16

        f = self.activation(x_t + I_t)

        x_tp1 = (x_t + self.dt * f * A_bf16) / (1 + self.dt * (1 / tau_bf16 + f))

        return x_tp1, x_tp1


class LTCN(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = jnp.bfloat16  

    @nn.compact
    def __call__(self, x, dt, *, train: bool = True):
        B, T, D = x.shape
        h0 = jnp.zeros((B, self.hidden_size), dtype=self.dtype)

        ltcn_scan = nn.scan(
            LTCNCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(
            hidden_size=self.hidden_size,
            dt=dt,
            compute_dtype=self.dtype,
            name="ltcn_cell",
            train=train
        )

        x_final, outputs = ltcn_scan(h0, x)
        return outputs

class SeqRegressor(nn.Module):
    features: int
    quantiles: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train: bool = True):
        out = nn.Dense(self.features * self.quantiles, dtype=self.dtype)(x)
        return out


class LTCNRegressor(nn.Module):
    features: int
    quantiles: int
    hidden_size: int
    dt: float
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train: bool = True):
        ltcn = LTCN(hidden_size=self.hidden_size, dtype=self.dtype)
        regressors = [SeqRegressor(1, self.quantiles, self.dtype) for _ in range(self.features)]

        x = ltcn(x, self.dt, train=train)

        out = jnp.stack([regressor(x[:, -1, :], train=train) for regressor in regressors], axis=1)
        return out.astype(jnp.float32)


class LTCNTrainState(train_state.TrainState):
    pass

def LTCNtrain_step(userloss):
    @jax.jit
    def func(state, batch, *args, **kwargs):
        """
        state: TrainState
        batch: dict with "x" and "y"
            x: (batch, time, input_features)
            y: (batch, features)
        """
        def loss_fn(params):
            preds = state.apply_fn(params, batch['x'])
            loss = userloss(
                preds, batch['y'], *args, **kwargs
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
    return func


def LTCNeval_step(userloss):
    @jax.jit
    def func(state, batch, *args, **kwargs):
        """
        state: TrainState
        batch: dict with "x" and "y"
            x: (batch, time, input_features)
            y: (batch, features)
        """
        def loss_fn(params):
            preds = state.apply_fn(params, batch['x'])
            loss = userloss(
                preds, batch['y'], *args, **kwargs,
            )
            return loss, preds
        
        loss, preds = loss_fn(state.params)
        loss = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name="batch"), loss
        )
        return loss, preds
    return func

    