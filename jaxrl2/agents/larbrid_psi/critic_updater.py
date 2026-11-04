from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_critic(
        key: PRNGKey, target_actor: TrainState, critic: TrainState,
        target_critic: TrainState, ddpm: TrainState, temp: TrainState,
        latents: jnp.ndarray, actions: jnp.ndarray, next_latents: jnp.ndarray,
        batch: DatasetDict, backup_entropy: bool = False,
        critic_reduction: str = 'min') -> Tuple[TrainState, Dict[str, float]]:
    features = jax.lax.stop_gradient(
        ddpm.apply_fn({'params': ddpm.params}, latents, actions, method='forward_phi'))

    dist = target_actor.apply_fn({'params': target_actor.params}, next_latents)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_features = ddpm.apply_fn({'params': ddpm.params}, next_latents, next_actions,
                                  method='forward_phi')
    next_qs = target_critic.apply_fn({'params': target_critic.params}, next_features)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch['rewards'] + batch['discount'] * batch['masks'] * next_q

    if backup_entropy:
        target_q -= batch['discount'] * batch['masks'] * temp.apply_fn(
            {'params': temp.params}) * next_log_probs
    target_q = jax.lax.stop_gradient(target_q)

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, features)
        critic_loss = 0.5 * ((qs - target_q) ** 2).mean(axis=-1).sum()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': qs.mean(),
            'target_actor_entropy': -next_log_probs.mean(),
            'next_log_probs': next_log_probs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
