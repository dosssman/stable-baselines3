#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

# export WANDB_DIR=~/random/rl/hwm-logs/hwm_minigrid_2
# if [ ! -d $WANDB_DIR ]; then
#   mkdir -p $WANDB_DIR
# fi

############################
# region: PPO Baselines FourRoom task (custom)
############################
export CUDA_VISIBLE_DEVICES=6
(sleep 1s && nohup xvfb-run -a python ppo_fact_state_4rooms.py) >& /dev/null &
# (sleep 1s && nohup xvfb-run -a python ppo_full_grid_obs_4rooms.py ) >& /dev/null &
############################
# endregion: PPO Baselines FourRoom task (custom)
############################

export CUDA_VISIBLE_DEVICES=

export WANDB_DIR=