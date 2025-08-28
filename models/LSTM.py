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

        # Wrap the OptimizedLSTMCell with scan across time
        lstm_scan = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,    # time axis in input
            out_axes=1    # keep time axis in outputs
        )(self.hidden_size, dtype=self.dtype, name="lstm_cell")

        (h_final, c_final), outputs = lstm_scan((h0, c0), x)
        return outputs

class SeqRegressor(nn.Module):
    features: int
    quantiles: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        """
        x: (batch, features)
        returns: (batch, features, quantiles)
        """
        out = nn.Dense(self.features * self.quantiles, dtype=self.dtype)(x)
        return out

class LSTMRegressor(nn.Module):
    features: int
    quantiles: int
    hidden_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        lstm = LSTM(hidden_size=self.hidden_size, dtype=self.dtype)
        regressors = [SeqRegressor(1, self.quantiles) for _ in range(self.features)]

        x = lstm(x)
        out = jnp.stack([regressor(x[:,-1,:]) for regressor in regressors],axis=1)

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