import argparse
import sys
from examples.train_sim import main
from jaxrl2.utils.launch_util import parse_training_args

# language instruction (pi0 prompt) for each supported libero_90 task id
LIBERO_90_TASK_LANGUAGE = {
    7: 'open the top drawer of the cabinet',
    20: 'turn on the stove',
    57: 'pick up the cream cheese and put it in the tray',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--task_id', default=57, help='libero_90 task id (libero only)', type=int)
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)
    parser.add_argument('--ae_warmup_steps', default=5000, help='number of autoencoder-only gradient steps before online updates (larbrid_psi, img_reconstruct=1 only)', type=int)

    pixel_sac_train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        hidden_dims= (128, 128, 128),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.999,
        tau= 0.005,
        critic_reduction = 'mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        )

    larbrid_psi_train_args_dict = dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        ddpm_lr=1e-4,
        ae_lr=1e-4,
        hidden_dims=(256, 256),
        discount=0.999,
        tau=0.005,
        critic_reduction='mean',
        backup_entropy=False,
        init_temperature=1.0,
        num_qs=10,
        target_entropy='auto',
        action_magnitude=1.0,
        num_cameras=1,
        stack_window=3, # number of consecutive observations [o_{t-2}, o_{t-1}, o_t] fed to the agent; 1 = single frame
        features_dim=128, # img embed dimension, per frame
        enc_num_filters=32,
        enc_num_layers=4,
        dec_num_filters=32,
        dec_num_layers=4,
        ae_recon_coef=1.0,
        ae_var_coef=0.1,
        ae_aug_pad=4,
        color_jitter=True,
        obs_target_encoder=False,
        img_reconstruct=False, # False: no autoencoder; the DDPM and the actor each own a pixel_sac encoder
        # pixel_sac encoder, only used when img_reconstruct=False
        encoder_type='resnet_small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        latent_dim=50, # img embed dimension, per frame
        aug_next=1,
        ddpm_embed_dim=128,
        ddpm_feature_dim=256,
        ddpm_phi_hidden_dims=(256, 256, 256, 256),
        ddpm_mu_hidden_dims=(256, 256, 256, 256),
        ddpm_reward_hidden_dims=(256,),
        ddpm_rff_dim=512,
        ddpm_sample_steps=50,
        )

    algorithm = parser.parse_known_args()[0].algorithm
    if algorithm == 'pixel_sac':
        train_args_dict = pixel_sac_train_args_dict
    elif algorithm == 'larbrid_psi':
        train_args_dict = larbrid_psi_train_args_dict
    else:
        raise NotImplementedError(algorithm)

    variant, args = parse_training_args(train_args_dict, parser)
    if variant.env == 'libero':
        if variant.task_id not in LIBERO_90_TASK_LANGUAGE:
            raise ValueError(f'language instruction is not set for libero_90 task id {variant.task_id}, '
                             f'supported ids: {sorted(LIBERO_90_TASK_LANGUAGE)}')
        variant.task_description = LIBERO_90_TASK_LANGUAGE[variant.task_id]
    print(variant)
    main(variant)
    sys.exit()
    