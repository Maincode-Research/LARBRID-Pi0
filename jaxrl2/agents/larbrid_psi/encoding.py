from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def split_images(pixels: jnp.ndarray, num_cameras: int) -> jnp.ndarray:
    """(B, H, W, 3 * num_cameras, T) -> (B * T * num_cameras, H, W, 3), ordered sample, frame, camera."""
    b, h, w, _, t = pixels.shape
    x = pixels.reshape(b, h, w, num_cameras, 3, t)
    x = jnp.transpose(x, (0, 5, 3, 1, 2, 4))
    return x.reshape(b * t * num_cameras, h, w, 3)


def layer_norm(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """LayerNorm over the last axis without learnable affine parameters."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def encode_obs(ae_apply_fn: Callable, encoder_params: Params, observations: DatasetDict,
               img_latent_scale: jnp.ndarray, num_cameras: int) -> jnp.ndarray:
    """Builds the (detached) latent state [LN(z(I)) * img_latent_scale, state] for every frame and camera."""
    pixels = observations['pixels']
    batch_size = pixels.shape[0]
    z = ae_apply_fn({'params': {'encoder': encoder_params}},
                    split_images(pixels, num_cameras),
                    method='encode')
    z = layer_norm(z) * img_latent_scale
    latent = z.reshape(batch_size, -1)
    if 'state' in observations:
        # (B, state_dim, T) -> (B, T * state_dim), frame-major like the image latents
        state = jnp.swapaxes(observations['state'], -1, -2).reshape(batch_size, -1).astype(jnp.float32)
        latent = jnp.concatenate([latent, state], axis=-1)
    return jax.lax.stop_gradient(latent)


def _color_jitter_newest(key: PRNGKey, pixels: jnp.ndarray, num_cameras: int) -> jnp.ndarray:
    """pixel_sac color jitter of the newest frame only, one jitter per sample shared by its cameras."""
    newest = pixels[..., -1:]
    for i in range(num_cameras):
        newest = newest.at[:, :, :, i * 3:(i + 1) * 3].set(
            (color_transform(key, newest[:, :, :, i * 3:(i + 1) * 3].astype(jnp.float32) / 255.) * 255
             ).astype(jnp.uint8))
    return pixels.at[..., -1:].set(newest)


def augment_window_batch(rng: PRNGKey, batch: DatasetDict, color_jitter: bool, aug_next: bool,
                         num_cameras: int) -> Tuple[PRNGKey, DatasetDict]:
    """pixel_sac augment_batch over a window of frames (B, H, W, C, T): one random crop per sample shared by
    all of its frames, and color jitter of the newest frame only. With a single frame this is exactly
    augment_batch (same random keys)."""
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']
    if aug_pixels.squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, aug_pixels)
        if color_jitter:
            rng, key = jax.random.split(rng)
            aug_pixels = _color_jitter_newest(key, aug_pixels, num_cameras)

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, aug_next_pixels)
        if color_jitter:
            rng, key = jax.random.split(rng)
            aug_next_pixels = _color_jitter_newest(key, aug_next_pixels, num_cameras)

    return rng, batch.copy(add_or_replace={
        'observations': batch['observations'].copy(add_or_replace={'pixels': aug_pixels}),
        'next_observations': batch['next_observations'].copy(add_or_replace={'pixels': aug_next_pixels}),
    })
