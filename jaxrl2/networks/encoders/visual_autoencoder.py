from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.constants import default_init


class VisualEncoder(nn.Module):
    """Single-frame conv encoder: (N, H, W, 3) uint8 images -> (N, features_dim) latents."""
    features_dim: int = 128
    num_filters: int = 32
    num_layers: int = 4

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = observations.astype(jnp.float32) / 255.0 - 0.5

        x = nn.Conv(self.num_filters, kernel_size=(3, 3), strides=(2, 2),
                    padding=((1, 1), (1, 1)), kernel_init=default_init())(x)
        x = nn.relu(x)
        for _ in range(self.num_layers - 1):
            x = nn.Conv(self.num_filters, kernel_size=(3, 3), strides=(1, 1),
                        padding=((1, 1), (1, 1)), kernel_init=default_init())(x)
            x = nn.relu(x)

        x = x.reshape((*x.shape[:-3], -1))
        x = nn.Dense(self.features_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dense(self.features_dim, kernel_init=default_init())(x)
        return x


class VisualDecoder(nn.Module):
    """Mirror of VisualEncoder: (N, latent_dim) -> (N, out_h, out_w, out_channels), unbounded."""
    enc_h: int
    enc_w: int
    out_h: int
    out_w: int
    out_channels: int = 3
    num_filters: int = 32
    num_layers: int = 4

    @nn.compact
    def __call__(self, latent: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.enc_h * self.enc_w * self.num_filters, kernel_init=default_init())(latent)
        x = nn.relu(x)
        x = x.reshape((*x.shape[:-1], self.enc_h, self.enc_w, self.num_filters))

        for _ in range(self.num_layers - 1):
            x = nn.ConvTranspose(self.num_filters, kernel_size=(3, 3), strides=(1, 1),
                                 padding=((1, 1), (1, 1)), kernel_init=default_init())(x)
            x = nn.relu(x)

        # Equivalent of torch ConvTranspose2d(k=3, s=2, p=1, output_padding=op): pad (k-1-p, k-1-p+op).
        op_h = 1 if self.out_h % 2 == 0 else 0
        op_w = 1 if self.out_w % 2 == 0 else 0
        x = nn.ConvTranspose(self.out_channels, kernel_size=(3, 3), strides=(2, 2),
                             padding=((1, 1 + op_h), (1, 1 + op_w)), kernel_init=default_init())(x)
        return x


class VisualAutoEncoder(nn.Module):
    """Image autoencoder with a shared encoder used for the RL latent state.

    image_shape is the (H, W, C) of a single frame of a single camera.
    """
    image_shape: Tuple[int, int, int]
    features_dim: int = 128
    enc_num_filters: int = 32
    enc_num_layers: int = 4
    dec_num_filters: int = 32
    dec_num_layers: int = 4

    def setup(self):
        h, w, c = self.image_shape
        self.encoder = VisualEncoder(self.features_dim, self.enc_num_filters, self.enc_num_layers)
        self.decoder = VisualDecoder(enc_h=(h - 1) // 2 + 1,
                                     enc_w=(w - 1) // 2 + 1,
                                     out_h=h,
                                     out_w=w,
                                     out_channels=c,
                                     num_filters=self.dec_num_filters,
                                     num_layers=self.dec_num_layers)

    def encode(self, images: jnp.ndarray) -> jnp.ndarray:
        return self.encoder(images)

    def decode(self, latent: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(latent)

    def __call__(self, images: jnp.ndarray, training: bool = False):
        z = self.encode(images)
        return z, self.decode(z)
