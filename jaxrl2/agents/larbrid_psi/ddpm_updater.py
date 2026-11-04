from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey


def update_ddpm(key: PRNGKey, ddpm: TrainState, latents: jnp.ndarray, actions: jnp.ndarray,
                next_latents: jnp.ndarray, rewards: jnp.ndarray,
                alphabars: jnp.ndarray) -> Tuple[TrainState, Dict[str, float]]:
    key_t, key_eps = jax.random.split(key)
    sample_steps = alphabars.shape[0]
    noise_idx = jax.random.randint(key_t, (next_latents.shape[0], ), 0, sample_steps)
    alphabar = alphabars[noise_idx][:, None]
    eps = jax.random.normal(key_eps, next_latents.shape)
    noisy_next_latents = jax.lax.stop_gradient(
        jnp.sqrt(alphabar) * next_latents + jnp.sqrt(1 - alphabar) * eps)

    def ddpm_loss_fn(ddpm_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        phi, mu, reward_pred = ddpm.apply_fn({'params': ddpm_params}, latents, actions,
                                             noisy_next_latents, noise_idx[:, None])
        eps_pred = jnp.einsum('bf,bfo->bo', phi, mu)
        diffusion_loss = ((eps_pred - eps) ** 2).sum(-1).mean()
        reward_loss = ((reward_pred.squeeze(-1) - rewards) ** 2).mean()
        ddpm_loss = diffusion_loss + reward_loss
        return ddpm_loss, {
            'ddpm_loss': ddpm_loss,
            'diffusion_loss': diffusion_loss,
            'reward_loss': reward_loss,
            'phi_norm': jnp.linalg.norm(phi, axis=-1).mean(),
        }

    grads, info = jax.grad(ddpm_loss_fn, has_aux=True)(ddpm.params)
    new_ddpm = ddpm.apply_gradients(grads=grads)

    return new_ddpm, info
