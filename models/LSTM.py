import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

class LSTM(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        """
        x: (batch, time, features)
        returns: (batch, time, hidden_size)
        """
        B, T, D = x.shape
        h0 = jnp.zeros((B, self.hidden_size))
        c0 = jnp.zeros((B, self.hidden_size))

        # Wrap the OptimizedLSTMCell with scan across time
        lstm_scan = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,    # time axis in input
            out_axes=1    # keep time axis in outputs
        )(self.hidden_size, name="lstm_cell")

        (h_final, c_final), outputs = lstm_scan((h0, c0), x)
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

class LSTMRegressor(nn.Module):
    features: int
    quantiles: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        lstm = LSTM(hidden_size=self.hidden_size)
        regressor = SeqRegressor(self.features, self.quantiles)

        x = lstm(x)
        out = regressor(x[:,-1,:])

        return out

class LSTMTrainState(train_state.TrainState):
    pass

@jax.jit
def quantile_loss(y_pred, y_true, quantiles):
    """
    y_pred: (Nstations, Nquantiles)
    y_true: (Nstations,) or (Nstations,1)
    quantiles: 1D array of quantiles, shape (Nquantiles,)
    """
    # Ensure y_true has shape (Nstations, 1)

    error = y_true - y_pred                     # (Nstations, Nquantiles)
    loss = jnp.maximum(quantiles * error, (quantiles - 1) * error)
    return jnp.mean(loss)

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
    mae = jnp.mean(jnp.abs(error))

    # Pinball loss
    loss = jnp.maximum(quantiles * error, (quantiles - 1) * error)  # (Nstations, Nquantiles)
    loss = jnp.sum(loss)
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
def LSTMtrain_step(state, batch, quantiles):
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
            crossing_penalty_coef=0.2, mae_coef=0.1
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
def LSTMeval_step(state, batch, quantiles):
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
            crossing_penalty_coef=0.2, mae_coef=0.1
        )
        return loss, preds
    
    loss, preds = loss_fn(state.params)
    loss = jax.tree_util.tree_map(
        lambda x: jax.lax.pmean(x, axis_name="batch"), loss
    )
    return loss, preds
    

if __name__ == "__main__":
    # Dummy data
    batch_size, time, input_features = 8, 20, 10
    features, hidden_size = 5, 32
    quantiles = jnp.array([0.1, 0.5, 0.9])

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, time, input_features))
    y = jax.random.normal(key, (batch_size, features))

    # Initialize model
    model = LSTMRegressor(features=features, quantiles=len(quantiles), hidden_size=hidden_size)
    variables = model.init(key, x)

    y_pred = model.apply(variables, x)
    print(y_pred.shape)