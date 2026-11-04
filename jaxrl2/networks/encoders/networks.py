from typing import Dict, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init, xavier_init, kaiming_init

from functools import partial
from typing import Any, Callable, Sequence, Tuple
import distrax

ModuleDef = Any

class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))
    

class PixelMultiplexer(nn.Module):
    encoder: Union[nn.Module, list]
    network: nn.Module
    latent_dim: int
    use_bottleneck: bool=True

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 training: bool = False):
        observations = FrozenDict(observations)

        x = self.encoder(observations['pixels'], training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        x = observations.copy(add_or_replace={'pixels': x})

        # print('fully connected keys', x.keys())
        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)


class PixelEncoder(nn.Module):
    """PixelMultiplexer's front end as a vector latent over a window of T frames:
    [z_1, ..., z_T, state_1, ..., state_T] with z = tanh(LN(Dense(encoder(pixels)))) per frame.

    pixels are (B, H, W, C, T) and state is (B, state_dim, T), oldest frame first. Each frame is encoded
    separately as its own batch element, so latent_dim is per frame; T = 1 is a single-frame latent.
    """
    encoder: nn.Module
    latent_dim: int
    use_bottleneck: bool = True

    @nn.compact
    def __call__(self, observations: Union[FrozenDict, Dict], training: bool = False) -> jnp.ndarray:
        observations = FrozenDict(observations)
        pixels = observations['pixels']
        batch_size, num_frames = pixels.shape[0], pixels.shape[-1]
        # (B, H, W, C, T) -> (B * T, H, W, C, 1), ordered sample, frame
        frames = jnp.moveaxis(pixels, -1, 1).reshape(batch_size * num_frames, *pixels.shape[1:-1], 1)
        x = self.encoder(frames, training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        x = x.reshape(batch_size, -1)
        if 'state' in observations:
            # (B, state_dim, T) -> (B, T * state_dim), frame-major
            state = jnp.swapaxes(observations['state'], -1, -2).reshape(batch_size, -1).astype(jnp.float32)
            x = jnp.concatenate([x, state], axis=-1)
        return x


class EncodedNetwork(nn.Module):
    """network(encoder(observations)), trained end to end."""
    encoder: nn.Module
    network: nn.Module

    @nn.compact
    def __call__(self, observations: Union[FrozenDict, Dict], training: bool = False):
        return self.network(self.encoder(observations, training), training=training)
