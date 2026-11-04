"""Diffusion spectral representation model.

See:
Shribak, D., Gao, C.X., Li, Y., Xiao, C. and Dai, B., 2024.
Diffusion spectral representation for reinforcement learning.
Advances in Neural Information Processing Systems, 37, pp.110028-110056.
"""
from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jaxrl2.networks.constants import default_init


def vp_noise_schedule(num_noises: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t = np.arange(1, num_noises + 1)
    b_max, b_min = 10., 0.1
    alphas = np.exp(-b_min / num_noises - 0.5 * (b_max - b_min) * (2 * t - 1) / num_noises ** 2)
    betas = 1 - alphas
    alphabars = np.cumprod(alphas, axis=0)
    return (jnp.asarray(betas, dtype=jnp.float32),
            jnp.asarray(alphas, dtype=jnp.float32),
            jnp.asarray(alphabars, dtype=jnp.float32))


class PositionalFeature(nn.Module):
    dim: int
    max_positions: int = 10000
    endpoint: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        freqs = jnp.arange(0, self.dim // 2, dtype=jnp.float32)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.astype(jnp.float32) @ freqs[None]
        return jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)


class ResidualMLPBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        if x.shape[-1] != self.hidden_dim:
            residual = nn.Dense(self.hidden_dim, kernel_init=default_init())(residual)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = jax.nn.mish(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        return x + residual


class ResidualMLP(nn.Module):
    output_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dims[0], kernel_init=default_init())(x)
        for size in self.hidden_dims:
            x = ResidualMLPBlock(size)(x)
        x = jax.nn.mish(x)
        x = nn.Dense(self.output_dim, kernel_init=default_init())(x)
        return x


class RffReward(nn.Module):
    """Reward prediction after a learnable random fourier feature layer."""
    hidden_dims: Sequence[int]
    rff_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.rff_dim // 2, kernel_init=default_init())(x)
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
        for size in self.hidden_dims:
            x = nn.Dense(size, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)
        x = nn.Dense(1, kernel_init=default_init())(x)
        return x


class DDPM(nn.Module):
    """Learns phi(s, a) and mu(s', t) such that phi^T mu predicts the noise added to s'."""
    obs_dim: int
    embed_dim: int = 256
    feature_dim: int = 512
    phi_hidden_dims: Sequence[int] = (1024, 1024, 1024, 1024)
    mu_hidden_dims: Sequence[int] = (1024, 1024, 1024, 1024)
    reward_hidden_dims: Sequence[int] = (1024,)
    rff_dim: int = 1024

    def setup(self):
        e = self.embed_dim
        self.mlp_s = nn.Sequential([nn.Dense(2 * e, kernel_init=default_init()), jax.nn.mish,
                                    nn.Dense(e, kernel_init=default_init())])
        self.mlp_a = nn.Sequential([nn.Dense(2 * e, kernel_init=default_init()), jax.nn.mish,
                                    nn.Dense(e, kernel_init=default_init())])
        self.mlp_t = nn.Sequential([PositionalFeature(e), nn.Dense(2 * e, kernel_init=default_init()),
                                    jax.nn.mish, nn.Dense(e, kernel_init=default_init())])
        self.mlp_phi = ResidualMLP(self.feature_dim, self.phi_hidden_dims)
        self.mlp_mu = ResidualMLP(self.obs_dim * self.feature_dim, self.mu_hidden_dims)
        self.reward_head = RffReward(self.reward_hidden_dims, self.rff_dim)

    def forward_phi(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([self.mlp_s(s), self.mlp_a(a)], axis=-1)
        return self.mlp_phi(x)

    def forward_mu(self, sp: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        t_ff = self.mlp_t(t)
        x = self.mlp_mu(jnp.concatenate([sp, t_ff], axis=-1))
        return x.reshape(-1, self.feature_dim, self.obs_dim)

    def reward(self, phi: jnp.ndarray) -> jnp.ndarray:
        return self.reward_head(phi)

    def __call__(self, s: jnp.ndarray, a: jnp.ndarray, sp: jnp.ndarray, t: jnp.ndarray):
        phi = self.forward_phi(s, a)
        mu = self.forward_mu(sp, t)
        return phi, mu, self.reward(phi)


class EncodedDDPM(nn.Module):
    """DDPM on top of its own pixel encoder: s = encoder(observations), trained by the DDPM loss.

    The next-state input sp of mu stays a latent (from a target copy of the encoder).
    """
    encoder: nn.Module
    ddpm: DDPM

    def encode(self, observations) -> jnp.ndarray:
        return self.encoder(observations)

    def forward_phi(self, observations, a: jnp.ndarray) -> jnp.ndarray:
        return self.ddpm.forward_phi(self.encoder(observations), a)

    def __call__(self, observations, a: jnp.ndarray, sp: jnp.ndarray, t: jnp.ndarray):
        return self.ddpm(self.encoder(observations), a, sp, t)
