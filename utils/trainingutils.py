import jax
import jax.numpy as jnp
from flax.training import train_state

@jax.jit
def cosine_annealing(step, base_lr, min_lr, steps_per_cycle, m_mul=0.95, t_mul=1.0):
    # Handle the special case t_mul == 1.0 separately (constant cycle length)
    def constant_cycle(step):
        cycle = jnp.floor(step / steps_per_cycle)
        step_in_cycle = step % steps_per_cycle
        peak_lr = base_lr * (m_mul ** cycle)
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * step_in_cycle / steps_per_cycle))
        return min_lr + (peak_lr - min_lr) * cosine_decay

    # General case for t_mul != 1.0
    def scaled_cycle(step):
        cycle = jnp.floor(
            jnp.log1p((t_mul - 1) * step / steps_per_cycle) / jnp.log(t_mul)
        )
        cycle = jnp.clip(cycle, 0.0, jnp.inf)

        steps_before = steps_per_cycle * (t_mul**cycle - 1) / (t_mul - 1)
        step_in_cycle = step - steps_before
        current_cycle_steps = steps_per_cycle * (t_mul**cycle)
        peak_lr = base_lr * (m_mul ** cycle)

        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * step_in_cycle / current_cycle_steps))
        return min_lr + (peak_lr - min_lr) * cosine_decay

    return jax.lax.cond(t_mul == 1.0, constant_cycle, scaled_cycle, step)


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
    horizon_weights=None,       # shape (Nhorizons,)
    crossing_penalty_coef=0.0,
    cov_weight=0.25,
    k=15, 
    mae_coef=0.5
):
    """
    Flexible quantile (pinball) loss with optional horizon and station weighting.
    """
    # Ensure y_true shape matches y_pred
    error = y_true - y_pred  # (Nstations, Nhorizons, Nquantiles)
    abs_e = jnp.abs(error)

    # Huberized error
    delta = 1.0
    huber_e = jnp.where(abs_e <= delta, 0.5 * error**2 / delta, abs_e - 0.5 * delta)
    
    # Pinball loss
    pinball = jnp.maximum(quantiles * huber_e, (quantiles - 1.0) * huber_e)

    # Apply horizon weights if provided
    if horizon_weights is not None:
        pinball = pinball * horizon_weights[None, :, None]

    loss = jnp.mean(pinball)

    # Crossing penalty across quantiles
    def compute_penalty(_):
        return jnp.mean(jnp.maximum(0, y_pred[:, :, :-1] - y_pred[:, :, 1:]))
    crossing_penalty = jax.lax.cond(crossing_penalty_coef > 0.0, compute_penalty, lambda _: 0.0, operand=None)

    # coverage loss
    
    # Differentiable indicator: (batch, n_horizons, n_quantiles)
    indicator_low = jax.nn.sigmoid(k * (y_true - y_pred[:,:,0, None]))
    indicator_high = jax.nn.sigmoid(k * (y_pred[:,:,-1, None] - y_true))
    
    indicator = indicator_high*indicator_low

    # extremum coverage
    cov = jnp.mean(indicator)

    # median absolute error 
    mae = jnp.mean(jnp.abs(y_true - y_pred[:, :, y_pred.shape[-1]//2, None]))

    total_loss = loss + crossing_penalty_coef * crossing_penalty + cov_weight * (cov - (quantiles.max() - quantiles.min()))**2 + mae_coef*mae

    return total_loss

def train_step(userloss):
    @jax.jit
    def func(state, batch, rng, *args, **kwargs):
        rng, dropout_key = jax.random.split(rng)
        
        def loss_fn(params):
            preds = state.apply_fn(params, batch['x'], train=True, rngs={'dropout': dropout_key})
            loss = userloss(preds, batch['y'], *args, **kwargs)
            return loss

        # Loss + grads
        loss, grads = jax.value_and_grad(loss_fn)(state.params)

        # pmean across devices
        #grads = jax.tree_util.tree_map(lambda g: jax.lax.pmean(g, axis_name="batch"), grads)
        #grads = jax.tree_util.tree_map(lambda g: jax.lax.psum(g, 'batch') / jax.device_count(), grads)
        #loss = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="batch"), loss)
        #loss = jax.tree_util.tree_map(lambda g: jax.lax.psum(g, 'batch') / jax.device_count(), loss)
        
        # Update state
        state = state.apply_gradients(grads=grads)
        return state, loss

    return func

def eval_step(userloss):
    @jax.jit
    def func(state, batch, *args, **kwargs):
        """
        state: TrainState
        batch: dict with "x" and "y"
            x: (batch, time, input_features)
            y: (batch, features)
        """
        # Forward pass without stochasticity
        preds = state.apply_fn(state.params, batch['x'], train=False)

        # Optionally cast to float32
        preds = preds.astype(jnp.float32)

        # Compute loss
        loss = userloss(preds, batch['y'], *args, **kwargs)

        # Average over devices
        #loss = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="batch"), loss)

        return loss, preds

    return func

class ModelTrainState(train_state.TrainState):
    pass

@jax.jit
def NLL(mu, logvar, y_true):
    """
    mu, logvar: (batch, nhorizon)
    y_true: (batch, nhorizon)
    """
    mu = mu[:,:,None]
    var = jnp.exp(logvar)[:,:,None]
    
    nll = 0.5 * (jnp.log(2 * jnp.pi * var) + (y_true - mu)**2 / var)
    return jnp.mean(nll)

@jax.jit
def ELBO_train_step(state, batch, rng, step, total_steps, *args, **kwargs):
    rng, dropout_key = jax.random.split(rng)

    def loss_fn(params):
        (mu, logvar), state_out = state.apply_fn(
            params, batch['x'], train=True,
            rngs={'dropout': dropout_key, 'sample': rng},
            mutable=['ELBO']
        )

        kl_list = jnp.array([k 
           for head in state_out['ELBO'].values() 
           for layer in head.values() 
           for k in layer['kl']])

        kl_total = jnp.sum(kl_list) / batch['x'].shape[0]
        
        beta = jnp.tanh(5 * step / total_steps)

        nll = NLL(mu, logvar, batch['y'], *args, **kwargs)
        loss = nll + beta * kl_total
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def ELBO_eval_step(state, batch, rng, step, total_steps, *args, **kwargs):
    rng, dropout_key = jax.random.split(rng)
    
    (mu, logvar), state_out = state.apply_fn(
        state.params, batch['x'], train=False,
        rngs={'sample': rng}, mutable=['ELBO']
    )
    kl_list = jnp.array([k 
        for head in state_out['ELBO'].values() 
        for layer in head.values() 
        for k in layer['kl']])

    kl_total = jnp.sum(kl_list) / batch['x'].shape[0]
    beta = jnp.tanh(5 * step / total_steps)
    nll = NLL(mu, logvar, batch['y'], *args, **kwargs)
    loss = nll + beta * kl_total
    return loss, mu, logvar

