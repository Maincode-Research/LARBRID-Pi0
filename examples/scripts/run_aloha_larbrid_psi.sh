#!/bin/bash
proj_name=DSRL_pi0_Aloha # same wandb project as run_aloha.sh; the --prefix below keeps a separate wandb group
device_id=0
seed=${SEED:-0} # override with SEED=<seed> bash <this script>
algorithm=larbrid_psi

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name;
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"  # repo root first, so `examples` resolves to this repo


# mujoco is pinned by the venv (3.3.1, see requirements.txt); 2.3.7 has no 3.12 wheel

# shared settings follow run_aloha.sh (pixel sac); larbrid_psi agent configs keep their defaults in launch_train_sim.py
python3 examples/launch_train_sim.py \
--algorithm ${algorithm} \
--env aloha_cube \
--prefix larbrid_pi0_aloha_${algorithm} \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed ${seed} \
--max_steps 500000  \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 1000 \
--resize_image 64 \
--action_magnitude 2.0 \
--query_freq 50 \
--target_entropy 0.0 \
--ae_warmup_steps 5000
