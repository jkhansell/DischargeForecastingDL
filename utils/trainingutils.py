import jax
import jax.numpy as jnp

@jax.jit
def cosine_annealing(step, base_lr, min_lr, steps_per_cycle, m_mul=0.95, t_mul=1.0):
    # compute cycle index
    cycle = jnp.floor(jnp.log1p((t_mul - 1) * step / steps_per_cycle) / jnp.log(t_mul))
    cycle = jnp.clip(cycle, 0.0, jnp.inf)  # safe
    steps_before = steps_per_cycle * (t_mul**cycle - 1) / (t_mul - 1)
    step_in_cycle = step - steps_before
    current_cycle_steps = steps_per_cycle * t_mul**cycle

    # decreasing peak
    peak_lr = base_lr * m_mul**cycle

    # cosine decay
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * step_in_cycle / current_cycle_steps))

    lr = min_lr + (peak_lr - min_lr) * cosine_decay
    return lr


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

# ---------- utilities ----------
def pairwise_sq_dist(x, y):
    """
    x: (H,), y: (H,)
    returns (H, H) matrix of (x_i - y_j)^2
    """
    return (x[:, None] - y[None, :]) ** 2

def soft_dtw_batched(D, gamma: float):
    """
    Batched soft-DTW cost.
    D: (B, T, T) non-negative pairwise cost matrices
    returns: (B,) soft-DTW costs
    """
    B, T, _ = D.shape
    inf = jnp.inf

    def one_batch(Di):
        # Di: (T, T)
        R = jnp.full((T + 1, T + 1), inf)
        R = R.at[0, 0].set(0.0)

        def row_body(i, R):
            def col_body(j, R):
                r0 = R[i, j]
                r1 = R[i, j + 1]
                r2 = R[i + 1, j]
                softmin = -gamma * jax.nn.logsumexp(
                    jnp.stack([-(r0 / gamma), -(r1 / gamma), -(r2 / gamma)])
                )
                R = R.at[i + 1, j + 1].set(Di[i, j] + softmin)
                return R
            R = jax.lax.fori_loop(0, T, col_body, R)
            return R

        R = jax.lax.fori_loop(0, T, row_body, R)
        return R[-1, -1]

    return jax.vmap(one_batch)(D)

def soft_dtw_expected_path(D, gamma: float):
    """
    Expected alignment path = gradient of soft-DTW wrt D.
    D: (T, T)
    returns: (T, T)
    """
    def cost_of_D(D_single):
        # Wrap to reuse batched implementation
        return soft_dtw_batched(D_single[None, ...], gamma=gamma)[0]
    return jax.grad(cost_of_D)(D)

# ---------- per-(H,d) losses ----------
def dilate_uni_losses(o_f: jnp.ndarray, t_f: jnp.ndarray, gamma: float):
    """
    o_f, t_f: (H,) — one feature’s predicted/true horizon
    returns: (shape_loss, temporal_loss) both scalars
    """
    H = o_f.shape[0]
    D = pairwise_sq_dist(t_f, o_f)               # (H, H)
    shape_loss = soft_dtw_batched(D[None, ...], gamma=gamma)[0]

    # Expected path from gradient wrt D
    path = soft_dtw_expected_path(D, gamma=gamma)  # (H, H)

    # Temporal distortion matrix Omega (index distances)
    idx = jnp.arange(H, dtype=jnp.float32)
    Omega = pairwise_sq_dist(idx, idx)            # (H, H)

    temporal_loss = jnp.sum(path * Omega) / (H * H)
    return shape_loss, temporal_loss

# ---------- multi-horizon, multivariate ----------
def dilate_loss(outputs: jnp.ndarray,
                targets: jnp.ndarray,
                alpha: float = 0.5,
                gamma: float = 0.01):
    """
    DILATE for multi-horizon forecasting.
    outputs, targets: (B, H, d)
    alpha in [0,1], gamma > 0
    returns: (loss, loss_shape, loss_temporal)
    """
    B, H, d = outputs.shape

    # vmap over features (axis=1 of (H,d) => feature axis is 1)
    # For each feature, uni-loss consumes (H,) vectors.
    feat_vmap = jax.vmap(lambda o, t: dilate_uni_losses(o, t, gamma),
                         in_axes=(1, 1),  # map over feature axis
                         out_axes=(0, 0)) # returns (d,), (d,)

    # vmap over batch
    batch_vmap = jax.vmap(feat_vmap, in_axes=(0, 0), out_axes=(0, 0))

    shape_by_bd, temporal_by_bd = batch_vmap(outputs[...,outputs.shape[-1]//2,None], targets)  # both (B, d)

    loss_shape = shape_by_bd.mean()
    loss_temporal = temporal_by_bd.mean()
    loss = alpha * loss_shape + (1.0 - alpha) * loss_temporal
    return jnp.sqrt(loss**2)