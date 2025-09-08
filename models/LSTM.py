import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

class LSTM(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        """
        x: (batch, time, features)
        returns: (batch, time, hidden_size)
        """
        B, T, D = x.shape
        h0 = jnp.zeros((B, self.hidden_size), dtype=self.dtype)
        c0 = jnp.zeros((B, self.hidden_size), dtype=self.dtype)

        lstm_scan = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1
        )(self.hidden_size, dtype=self.dtype, name="lstm_cell")

        (h_final, c_final), outputs = lstm_scan((h0, c0), x)
        return outputs


# ------------------------------
# Seq Regressor
# ------------------------------
class SeqRegressor(nn.Module):
    quantiles: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):

        x = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.relu,
            nn.Dense(self.hidden_size),
            nn.relu,
        ])(x)
    
        out = nn.Dense(self.quantiles)(x)
        return out.astype(jnp.float32)  # final output float32


class LSTMRegressor(nn.Module):
    features: int
    quantiles: int
    hidden_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train: bool = True):
        lstm = LSTM(hidden_size=self.hidden_size, dtype=self.dtype)
        regressors = [SeqRegressor(quantiles=self.n_quantiles, hidden_size=self.d_model) for _ in range(self.out_features)]

        x = lstm(x)
        out = jnp.stack([regressor(x[:, -1, :]) for regressor in regressors], axis=1)

        return out.astype(jnp.float32)

class LSTMTrainState(train_state.TrainState):
    pass

def LSTMtrain_step(userloss):
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


def LSTMeval_step(userloss):
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