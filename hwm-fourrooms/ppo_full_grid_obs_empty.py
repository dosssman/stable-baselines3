import gym

import numpy as np
import gym_minigrid
import hwm.gym_minigrid_2.fourroom_cstm # custom FourRoom envs
from gym_minigrid.wrappers import ReseedWrapper, ImgObsWrapper
from hwm.gym_minigrid_2.wrappers import RGBImgFullGridWrapper, ChannelFirstImgWrapper, \
    RGBImgResizeWrapper, ActionMaskingWrapper

from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

import os
from typing import Any, Callable, Dict, Optional, Type, Union
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
def make_vec_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            env = gym.make(env_id, **env_kwargs)
            
            if not wrapper_kwargs["no_reseed"]:
                env = ReseedWrapper(env, seeds=wrapper_kwargs["env_seeds"])
            else:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)

            env = RGBImgFullGridWrapper(env)
            env = RGBImgResizeWrapper(env, image_size=wrapper_kwargs["env_img_size"])
            env = ImgObsWrapper(env)
            env = ChannelFirstImgWrapper(env)
            
            if len(wrapper_kwargs["env_masked_actions"]):
                env = ActionMaskingWrapper(env, invalid_actions_list=wrapper_kwargs["env_masked_actions"])
            
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)

# Policy customization
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Scale(nn.Module):
    def __init__(self, old_min=0., old_max=255., new_min=-1., new_max=1.):
        super().__init__()
        self.old_min, self.old_max, self.new_min, self.new_max = \
            old_min, old_max, new_min, new_max

    def _scale_img(self, a):
        return ((a - self.old_min) * (self.new_max - self.new_min)) / (self.old_max - self.old_min) + self.new_min
    
    def forward(self, x):
        return self._scale_img(x)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # TODO: make sure the scale() is actually needed, and that the range is valid
        self.model = nn.Sequential(
            # Scale(),
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, features_dim)),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.model(observations)

# Parallel environments
# ENV_NAME = "MiniGrid-FourRooms-Size15-v0"
ENV_NAME = "MiniGrid-Empty-11x11-v0"

wrapper_kwargs = {
    "no_reseed": False,
    "env_seeds": [222],
    "env_masked_actions": ["pickup", "drop", "toggle", "done"],
    "env_img_size": [84,84],
}
env = make_vec_env(ENV_NAME, n_envs=4, wrapper_kwargs=wrapper_kwargs)

policy_kwargs = {
    "features_extractor_class": CustomCNN,
    "features_extractor_kwargs": {"features_dim": 512}
}

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=10_000_000, tb_log_name=f"PPO_CNN_FullImgObs_{ENV_NAME}")
model.save(f"models/PPO_CNN_FullImgObs_{ENV_NAME}_Agent")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()