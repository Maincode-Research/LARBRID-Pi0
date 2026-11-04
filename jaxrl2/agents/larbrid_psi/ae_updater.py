from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.agents.larbrid_psi.encoding import split_images
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_ae(key: PRNGKey, ae: TrainState, observations: DatasetDict, num_cameras: int, aug_pad: int,
              recon_coef: float, var_coef: float, color_jitter: bool) -> Tuple[TrainState, Dict[str, float]]:
    # only the most recent frame is reconstructed; the target is the un-augmented image
    images = split_images(observations['pixels'][..., -1:], num_cameras)
    target = images.astype(jnp.float32) / 255.0 - 0.5
    key_crop, key_jitter = jax.random.split(key)
    aug_images = batched_random_crop(key_crop, images[..., None], padding=aug_pad)
    if color_jitter:
        aug_images = (color_transform(key_jitter, aug_images.astype(jnp.float32) / 255.) * 255).astype(jnp.uint8)
    aug_images = aug_images[..., 0]

    def ae_loss_fn(ae_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        z, recon = ae.apply_fn({'params': ae_params}, aug_images)
        recon_loss = ((recon - target) ** 2).sum() / recon.shape[0]
        latent_std = jnp.std(z, axis=0, ddof=1)
        var_loss = jnp.maximum(1.0 - latent_std, 0.0).sum()
        ae_loss = recon_coef * recon_loss + var_coef * var_loss
        return ae_loss, {
            'ae_loss': ae_loss,
            'ae_recon_loss': recon_loss,
            'ae_var_loss': var_loss,
            'ae_latent_std': latent_std.mean(),
        }

    grads, info = jax.grad(ae_loss_fn, has_aux=True)(ae.params)
    new_ae = ae.apply_gradients(grads=grads)

    return new_ae, info
