import jax
import jax.numpy as jnp

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
    crossing_penalty_coef=0.0,
):
    """
    Flexible quantile (pinball) loss using JAX-lax conditionals for JIT stability.
    """

    # Ensure y_true is (Nstations, 1)

    # Error across all quantiles
    error = y_true - y_pred  # (Nstations, Nquantiles)

    # Pinball loss
    loss = jnp.maximum(quantiles * error, (quantiles - 1) * error)  # (Nstations, Nquantiles)
    loss = jnp.mean(loss)
    # Crossing penalty
    def compute_penalty(_):
        return jnp.mean(jnp.maximum(0, y_pred[:, :-1] - y_pred[:, 1:]))
    crossing_penalty = jax.lax.cond(crossing_penalty_coef > 0.0, compute_penalty, lambda _: jnp.bfloat16(0.0), operand=None)

    # Combine
    total_loss = loss + crossing_penalty_coef * crossing_penalty

    # Return either per-quantile or total
    return total_loss