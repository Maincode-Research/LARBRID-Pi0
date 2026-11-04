"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib

import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.larbrid_psi.actor_updater import update_actor
from jaxrl2.agents.larbrid_psi.ae_updater import update_ae
from jaxrl2.agents.larbrid_psi.critic_updater import update_critic
from jaxrl2.agents.larbrid_psi.ddpm_updater import update_ddpm
from jaxrl2.agents.larbrid_psi.encoding import augment_window_batch, encode_obs, split_images
from jaxrl2.agents.larbrid_psi.temperature_updater import update_temperature
from jaxrl2.agents.pixel_sac.pixel_sac_learner import make_encoder, make_visual
from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.ddpm import DDPM, EncodedDDPM, vp_noise_schedule
from jaxrl2.networks.encoders.networks import EncodedNetwork, PixelEncoder
from jaxrl2.networks.encoders.visual_autoencoder import VisualAutoEncoder
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import FeatureCriticEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


@functools.partial(jax.jit, static_argnames=('critic_reduction', 'num_cameras', 'obs_target_encoder', 'ae_aug_pad',
                                             'color_jitter', 'backup_entropy'))
def _update_jit(
    rng: PRNGKey, ae: TrainState, target_encoder_params: Params, ddpm: TrainState,
    actor: TrainState, target_actor_params: Params, critic: TrainState,
    target_critic_params: Params, temp: TrainState, batch: DatasetDict,
    img_latent_scale: jnp.ndarray, alphabars: jnp.ndarray, tau: float, target_entropy: float,
    ae_recon_coef: float, ae_var_coef: float,
    critic_reduction: str, num_cameras: int, obs_target_encoder: bool, ae_aug_pad: int,
    color_jitter: bool, backup_entropy: bool,
):
    # 1) autoencoder; the encoder is trained by the reconstruction loss only
    rng, key = jax.random.split(rng)
    new_ae, ae_info = update_ae(key, ae, batch['next_observations'], num_cameras, ae_aug_pad,
                                ae_recon_coef, ae_var_coef, color_jitter)

    # 2) detached latents (encode_obs stops gradients, so no loss below reaches either encoder)
    encode = functools.partial(encode_obs, new_ae.apply_fn, img_latent_scale=img_latent_scale,
                               num_cameras=num_cameras)
    online_encoder_params = new_ae.params['encoder']
    obs_encoder_params = target_encoder_params if obs_target_encoder else online_encoder_params
    latents = encode(obs_encoder_params, batch['observations'])
    # target encoder for the next state: DDPM (mu) regression target and critic backup
    next_latents = encode(target_encoder_params, batch['next_observations'])
    actions = batch['actions'].reshape(batch['actions'].shape[0], -1)

    # 3) temperature, with the pre-update alpha used by every loss below
    rng, key_act = jax.random.split(rng)
    dist = actor.apply_fn({'params': actor.params}, latents)
    _, log_probs = dist.sample_and_log_prob(seed=key_act)
    new_temp, alpha_info = update_temperature(temp, -log_probs.mean(), target_entropy)

    # 4) DDPM representation
    rng, key = jax.random.split(rng)
    new_ddpm, ddpm_info = update_ddpm(key, ddpm, latents, actions, next_latents,
                                      batch['rewards'], alphabars)

    # 5) critic on the pre-update DDPM features, bootstrapped with the target actor
    rng, key = jax.random.split(rng)
    target_actor = actor.replace(params=target_actor_params)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(key, target_actor, critic, target_critic, ddpm, temp,
                                            latents, actions, next_latents, batch,
                                            backup_entropy=backup_entropy,
                                            critic_reduction=critic_reduction)

    # 6) actor through the updated DDPM features and critic
    new_actor, actor_info = update_actor(key_act, actor, new_critic, new_ddpm, temp, latents,
                                         critic_reduction=critic_reduction)

    # 7) target networks
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)
    new_target_actor_params = soft_target_update(new_actor.params, target_actor_params, tau)
    new_target_encoder_params = soft_target_update(new_ae.params['encoder'], target_encoder_params, tau)

    return rng, new_ae, new_target_encoder_params, new_ddpm, new_actor, new_target_actor_params, \
        new_critic, new_target_critic_params, new_temp, {
            **ae_info,
            **ddpm_info,
            **critic_info,
            **actor_info,
            **alpha_info,
        }


@functools.partial(jax.jit, static_argnames=('critic_reduction', 'num_cameras', 'color_jitter', 'aug_next',
                                             'backup_entropy'))
def _update_pixel_jit(
    rng: PRNGKey, ddpm: TrainState, actor: TrainState,
    critic: TrainState, target_critic_params: Params, temp: TrainState,
    batch: DatasetDict, alphabars: jnp.ndarray, tau: float, target_entropy: float,
    critic_reduction: str, num_cameras: int, color_jitter: bool, aug_next: bool, backup_entropy: bool,
):
    """img_reconstruct=False: the DDPM and the actor each own a pixel_sac encoder (the critic reads the DDPM's).

    Only the critic has a target copy: as in pixel_sac the current actor bootstraps the critic, and the
    pre-update online DDPM provides both the regression target and the backup features.
    """
    # 1) pixel_sac augmentation (one crop per frame window, color jitter on the newest frame), shared below
    rng, batch = augment_window_batch(rng, batch, color_jitter, aug_next, num_cameras)
    observations = batch['observations']
    next_observations = batch['next_observations']
    actions = batch['actions'].reshape(batch['actions'].shape[0], -1)

    # 2) DDPM and its encoder; the noised regression target is the pre-update encoder's next-state latent
    next_latents = jax.lax.stop_gradient(
        ddpm.apply_fn({'params': ddpm.params}, next_observations, method='encode'))
    rng, key = jax.random.split(rng)
    new_ddpm, ddpm_info = update_ddpm(key, ddpm, observations, actions, next_latents,
                                      batch['rewards'], alphabars)

    # 3) critic on the pre-update DDPM features; the backup uses the current actor and the pre-update DDPM
    rng, key = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(key, actor, critic, target_critic, ddpm, temp,
                                            observations, actions, next_observations, batch,
                                            backup_entropy=backup_entropy,
                                            critic_reduction=critic_reduction)

    # 4) actor and its encoder, through the updated DDPM features and critic
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, new_ddpm, temp, observations,
                                         critic_reduction=critic_reduction)

    # 5) temperature from the pre-update actor's entropy (the losses above used the pre-update alpha)
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    # 6) target critic
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)

    return rng, new_ddpm, new_actor, \
        new_critic, new_target_critic_params, new_temp, {
            **ddpm_info,
            'ddpm_target_latent_std': next_latents.std(axis=0).mean(),
            **critic_info,
            **actor_info,
            **alpha_info,
        }


@functools.partial(jax.jit, static_argnames=('num_cameras', 'ae_aug_pad', 'color_jitter'))
def _update_ae_jit(rng: PRNGKey, ae: TrainState, batch: DatasetDict, ae_recon_coef: float,
                   ae_var_coef: float, num_cameras: int, ae_aug_pad: int, color_jitter: bool):
    rng, key = jax.random.split(rng)
    new_ae, info = update_ae(key, ae, batch['next_observations'], num_cameras, ae_aug_pad,
                             ae_recon_coef, ae_var_coef, color_jitter)
    return rng, new_ae, info


@functools.partial(jax.jit, static_argnames=('num_cameras', 'deterministic'))
def _actions_jit(rng: PRNGKey, ae: TrainState, encoder_params: Params, actor: TrainState,
                 observations: DatasetDict, img_latent_scale: jnp.ndarray, num_cameras: int,
                 deterministic: bool):
    latents = encode_obs(ae.apply_fn, encoder_params, observations, img_latent_scale, num_cameras)
    dist = actor.apply_fn({'params': actor.params}, latents)
    if deterministic:
        return rng, dist.mode()
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


@functools.partial(jax.jit, static_argnames=('num_cameras',))
def get_value(ae: TrainState, encoder_params: Params, ddpm: TrainState, critic: TrainState,
              observations: DatasetDict, actions: jnp.ndarray, img_latent_scale: jnp.ndarray,
              num_cameras: int):
    latents = encode_obs(ae.apply_fn, encoder_params, observations, img_latent_scale, num_cameras)
    actions = actions.reshape(actions.shape[0], -1)
    features = ddpm.apply_fn({'params': ddpm.params}, latents, actions, method='forward_phi')
    return critic.apply_fn({'params': critic.params}, features)


@functools.partial(jax.jit, static_argnames=('deterministic',))
def _pixel_actions_jit(rng: PRNGKey, actor: TrainState, observations: DatasetDict, deterministic: bool):
    dist = actor.apply_fn({'params': actor.params}, observations)
    if deterministic:
        return rng, dist.mode()
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


@jax.jit
def _pixel_get_value(ddpm: TrainState, critic: TrainState, observations: DatasetDict, actions: jnp.ndarray):
    actions = actions.reshape(actions.shape[0], -1)
    features = ddpm.apply_fn({'params': ddpm.params}, observations, actions, method='forward_phi')
    return critic.apply_fn({'params': critic.params}, features)


class LarbridPsiLearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 ddpm_lr: float = 1e-4,
                 ae_lr: float = 1e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (1024, 1024),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 critic_reduction: str = 'min',
                 backup_entropy: bool = False,
                 init_temperature: float = 1.0,
                 num_qs: int = 2,
                 target_entropy: float = None,
                 action_magnitude: float = 1.0,
                 num_cameras: int = 1,
                 stack_window: int = 1,
                 features_dim: int = 128,
                 enc_num_filters: int = 32,
                 enc_num_layers: int = 4,
                 dec_num_filters: int = 32,
                 dec_num_layers: int = 4,
                 ae_recon_coef: float = 1.0,
                 ae_var_coef: float = 0.1,
                 ae_aug_pad: int = 4,
                 color_jitter: bool = True,
                 obs_target_encoder: bool = False,
                 img_reconstruct: bool = True,
                 encoder_type: str = 'resnet_small',
                 encoder_norm: str = 'group',
                 use_spatial_softmax: bool = True,
                 softmax_temperature: float = -1,
                 latent_dim: int = 50,
                 aug_next: bool = True,
                 ddpm_embed_dim: int = 256,
                 ddpm_feature_dim: int = 512,
                 ddpm_phi_hidden_dims: Sequence[int] = (1024, 1024, 1024, 1024),
                 ddpm_mu_hidden_dims: Sequence[int] = (1024, 1024, 1024, 1024),
                 ddpm_reward_hidden_dims: Sequence[int] = (1024,),
                 ddpm_rff_dim: int = 1024,
                 ddpm_sample_steps: int = 50,
                 ):
        """
        Soft-Actor-Critic in the noise space of a diffusion policy, with a critic on top of a
        diffusion spectral representation phi(s, a) of an image-autoencoder latent state s.

        img_reconstruct=False drops the autoencoder: the DDPM (whose phi the critic reads) and the actor
        each own a pixel_sac encoder + bottleneck, trained by their own losses. Only the critic has a
        target copy; as in pixel_sac there is no target actor.

        stack_window: number of consecutive decision-step observations [o_{t-W+1}, ..., o_t] stacked on the
        trailing frame axis of the pixels and state. Each frame is encoded separately (latent_dim and
        features_dim are per frame) and the latent state is [z_{t-W+1}, ..., z_t, q_{t-W+1}, ..., q_t].
        """

        self.img_reconstruct = bool(img_reconstruct)
        self.aug_next = bool(aug_next)
        self.num_cameras = num_cameras
        self.stack_window = int(stack_window)
        assert observations['pixels'].shape[-1] == self.stack_window, \
            f"observations stack {observations['pixels'].shape[-1]} frames, stack_window={self.stack_window}"
        self.ae_recon_coef = ae_recon_coef
        self.ae_var_coef = ae_var_coef
        self.ae_aug_pad = ae_aug_pad
        self.color_jitter = bool(color_jitter)
        self.backup_entropy = bool(backup_entropy)
        self.obs_target_encoder = bool(obs_target_encoder)

        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction

        rng = jax.random.PRNGKey(seed)
        rng, ae_key, ddpm_key, actor_key, critic_key, temp_key = jax.random.split(rng, 6)

        sample_actions = jnp.asarray(actions, dtype=jnp.float32).reshape(actions.shape[0], -1)
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0])

        policy_def = LearnedStdTanhNormalPolicy(hidden_dims, self.action_dim,
                                                low=-action_magnitude, high=action_magnitude,
                                                activations=nn.tanh, use_layer_norm=True)

        if self.img_reconstruct:
            # image autoencoder; its encoder provides the latent state for the DDPM, critic and actor
            image_shape = observations['pixels'].shape[1:3] + (3, )
            ae_def = VisualAutoEncoder(image_shape=image_shape,
                                       features_dim=features_dim,
                                       enc_num_filters=enc_num_filters,
                                       enc_num_layers=enc_num_layers,
                                       dec_num_filters=dec_num_filters,
                                       dec_num_layers=dec_num_layers)
            print(ae_def)
            ae_params = ae_def.init(ae_key, split_images(observations['pixels'][..., -1:], num_cameras))['params']
            ae = TrainState.create(apply_fn=ae_def.apply,
                                   params=ae_params,
                                   tx=optax.adam(learning_rate=ae_lr))
            target_encoder_params = copy.deepcopy(ae_params['encoder'])
            img_latent_scale = jnp.ones((), dtype=jnp.float32)

            sample_latents = encode_obs(ae.apply_fn, ae_params['encoder'], observations, img_latent_scale, num_cameras)
            # the DDPM and the actor take the latent state
            sample_inputs = sample_latents
        else:
            # no autoencoder: the DDPM and the actor each own a pixel_sac encoder, fed the raw observations
            if encoder_norm in ('batch', 'cross'):
                raise ValueError(f'encoder_norm={encoder_norm} keeps batch_stats, which larbrid_psi does not carry')
            ae, target_encoder_params, img_latent_scale = None, None, None
            encoder_def = make_encoder(encoder_type, encoder_norm, bool(use_spatial_softmax), softmax_temperature)
            pixel_encoder_def = PixelEncoder(encoder=encoder_def, latent_dim=latent_dim)
            sample_latents = pixel_encoder_def.apply(pixel_encoder_def.init(ae_key, observations), observations)
            sample_inputs = observations

        obs_dim = sample_latents.shape[-1]
        print(f'latent state dim: {obs_dim}')

        ddpm_def = DDPM(obs_dim=obs_dim,
                        embed_dim=ddpm_embed_dim,
                        feature_dim=ddpm_feature_dim,
                        phi_hidden_dims=tuple(ddpm_phi_hidden_dims),
                        mu_hidden_dims=tuple(ddpm_mu_hidden_dims),
                        reward_hidden_dims=tuple(ddpm_reward_hidden_dims),
                        rff_dim=ddpm_rff_dim)
        actor_def = policy_def
        if not self.img_reconstruct:
            ddpm_def = EncodedDDPM(encoder=pixel_encoder_def, ddpm=ddpm_def)
            actor_def = EncodedNetwork(encoder=pixel_encoder_def, network=policy_def)

        print(ddpm_def)
        ddpm_params = ddpm_def.init(ddpm_key, sample_inputs, sample_actions, sample_latents,
                                    jnp.zeros((sample_latents.shape[0], 1), dtype=jnp.int32))['params']
        ddpm = TrainState.create(apply_fn=ddpm_def.apply,
                                 params=ddpm_params,
                                 tx=optax.adam(learning_rate=ddpm_lr))
        self._alphabars = vp_noise_schedule(ddpm_sample_steps)[2]

        print(actor_def)
        actor_params = actor_def.init(actor_key, sample_inputs)['params']
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr))
        # only used with the autoencoder; without it the current actor also bootstraps the critic
        target_actor_params = copy.deepcopy(actor_params) if self.img_reconstruct else None

        critic_def = FeatureCriticEnsemble(hidden_dims, num_qs=num_qs)
        print(critic_def)
        sample_features = ddpm_def.apply({'params': ddpm_params}, sample_inputs, sample_actions,
                                         method='forward_phi')
        critic_params = critic_def.init(critic_key, sample_features)['params']
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.adam(learning_rate=critic_lr))
        target_critic_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr))

        self._rng = rng
        self._ae = ae
        self._target_encoder_params = target_encoder_params
        self._img_latent_scale = img_latent_scale
        self._ddpm = ddpm
        self._actor = actor
        self._target_actor_params = target_actor_params
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        if target_entropy is None or target_entropy == 'auto':
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)
        print(f'target_entropy: {self.target_entropy}')
        print(self.critic_reduction)

    @property
    def _acting_encoder_params(self) -> Params:
        return self._target_encoder_params if self.obs_target_encoder else self._ae.params['encoder']

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        if not self.img_reconstruct:
            return self._update_pixel(batch)
        (new_rng, new_ae, new_target_encoder_params, new_ddpm, new_actor, new_target_actor_params,
         new_critic, new_target_critic_params, new_temp, info) = _update_jit(
            self._rng, self._ae, self._target_encoder_params, self._ddpm, self._actor,
            self._target_actor_params, self._critic, self._target_critic_params, self._temp, batch,
            self._img_latent_scale, self._alphabars, self.tau, self.target_entropy,
            self.ae_recon_coef, self.ae_var_coef,
            self.critic_reduction, self.num_cameras, self.obs_target_encoder, self.ae_aug_pad,
            self.color_jitter, self.backup_entropy
            )

        self._rng = new_rng
        self._ae = new_ae
        self._target_encoder_params = new_target_encoder_params
        self._ddpm = new_ddpm
        self._actor = new_actor
        self._target_actor_params = new_target_actor_params
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        self._temp = new_temp
        return info

    def _update_pixel(self, batch: FrozenDict) -> Dict[str, float]:
        (new_rng, new_ddpm, new_actor,
         new_critic, new_target_critic_params, new_temp, info) = _update_pixel_jit(
            self._rng, self._ddpm, self._actor,
            self._critic, self._target_critic_params, self._temp, batch,
            self._alphabars, self.tau, self.target_entropy,
            self.critic_reduction, self.num_cameras, self.color_jitter, self.aug_next, self.backup_entropy
            )

        self._rng = new_rng
        self._ddpm = new_ddpm
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        self._temp = new_temp
        return info

    # --- autoencoder warm-up (img_reconstruct only), run once by the training loop before the first RL update ---
    def calibrate_latent_scale(self, batch: FrozenDict) -> float:
        """Scales the layer-normed image latent to the average per-dim std of the proprio state."""
        if 'state' in batch['observations']:
            state = np.asarray(batch['observations']['state'])
            state = state.reshape(state.shape[0], -1)
            self._img_latent_scale = jnp.asarray(state.std(axis=0, ddof=1).mean(), dtype=jnp.float32)
        return float(self._img_latent_scale)

    def update_ae(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_ae, info = _update_ae_jit(self._rng, self._ae, batch, self.ae_recon_coef,
                                               self.ae_var_coef, self.num_cameras, self.ae_aug_pad,
                                               self.color_jitter)
        self._rng = new_rng
        self._ae = new_ae
        return info

    def sync_target_encoder(self):
        self._target_encoder_params = self._ae.params['encoder']

    # --- acting ---
    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        if not self.img_reconstruct:
            rng, actions = _pixel_actions_jit(self._rng, self._actor, observations, False)
            self._rng = rng
            return np.asarray(actions)
        rng, actions = _actions_jit(self._rng, self._ae, self._acting_encoder_params, self._actor,
                                    observations, self._img_latent_scale, self.num_cameras, False)
        self._rng = rng
        return np.asarray(actions)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        if not self.img_reconstruct:
            _, actions = _pixel_actions_jit(self._rng, self._actor, observations, True)
            return np.asarray(actions)
        _, actions = _actions_jit(self._rng, self._ae, self._acting_encoder_params, self._actor,
                                  observations, self._img_latent_scale, self.num_cameras, True)
        return np.asarray(actions)

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        from examples.train_utils_sim import make_multiple_value_reward_visulizations
        make_multiple_value_reward_visulizations(self, variant, i, eval_buffer, wandb_logger)

    def make_value_reward_visulization(self, variant, trajs):
        num_traj = len(trajs['rewards'])
        traj_images = []

        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            if self.img_reconstruct:
                qs = get_value(self._ae, self._acting_encoder_params, self._ddpm, self._critic,
                               observations, actions, self._img_latent_scale, self.num_cameras)
            else:
                qs = _pixel_get_value(self._ddpm, self._critic, observations, actions)
            q_pred = list(np.asarray(qs).T)

            traj_images.append(make_visual(q_pred, rewards, masks, observations['pixels']))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        if self.img_reconstruct:
            mode_dict = {
                'ae': self._ae,
                'target_encoder_params': self._target_encoder_params,
                'img_latent_scale': self._img_latent_scale,
                'target_actor_params': self._target_actor_params,
            }
        else:
            mode_dict = {}
        save_dict = {
            **mode_dict,
            'ddpm': self._ddpm,
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'temp': self._temp
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        if self.img_reconstruct:
            self._ae = output_dict['ae']
            self._target_encoder_params = output_dict['target_encoder_params']
            self._img_latent_scale = jnp.asarray(output_dict['img_latent_scale'], dtype=jnp.float32)
            self._target_actor_params = output_dict['target_actor_params']
        self._ddpm = output_dict['ddpm']
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._temp = output_dict['temp']
        print('restored from ', dir)
