from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.mlp import MLP


class FeatureCritic(nn.Module):
    """Q-network on a learned state-action feature (no separate action input)."""
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh

    @nn.compact
    def __call__(self, features: jnp.ndarray, training: bool = False):
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations,
                     use_layer_norm=True)(features, training=training)
        return jnp.squeeze(critic, -1)


class FeatureCriticEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    num_qs: int = 2

    @nn.compact
    def __call__(self, features, training: bool = False):
        VmapCritic = nn.vmap(FeatureCritic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims, activations=self.activations)(features, training)
        return qs
