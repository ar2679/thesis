### Strategies
import numpy as np
import tensorflow as tf
from collections import namedtuple
from config import config
from typing import Type
from datetime import datetime
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import FeedForwardPolicy, LstmPolicy
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from functools import partial

Strategy = namedtuple('strategy', 'model model_name timesteps')


def train_A2C(env_train,
              model_name: str,
              seed: int,
              policy_kwargs: dict,
              timesteps: int = 25000):
    """A2C model"""
    start = datetime.now()
    model = A2C(
        'MlpPolicy', env_train, seed=seed, policy_kwargs=policy_kwargs, verbose=0
    )
    model.learn(total_timesteps=timesteps)
    end = datetime.now()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_DDPG(env_train,
               model_name: str,
               policy_kwargs: dict,
               seed: int = 42,
               timesteps: int = 10000):
    """DDPG model"""
    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions)
    )

    start = datetime.now()
    model = DDPG(
        'MlpPolicy', env_train, param_noise=param_noise,
        policy_kwargs=policy_kwargs, action_noise=action_noise, seed=seed
    )
    model.learn(total_timesteps=timesteps)
    end = datetime.now()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model


def train_PPO(env_train,
              model_name: str,
              policy_kwargs: dict,
              seed: int = 42,
              timesteps: int = 50000):
    """PPO model"""

    start = datetime.now()
    model = PPO2(
        'MlpPolicy', env_train, policy_kwargs=policy_kwargs,
        ent_coef = 0.005, nminibatches = 8, seed=seed
    )

    model.learn(total_timesteps=timesteps)
    end = datetime.now()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


# setting up the various architecture and activation function combinations
mlp_wider_relu = dict(layers=[128, 128], act_fun=tf.nn.relu)
mlp_default_relu = dict(layers=[64, 64], act_fun=tf.nn.relu)
mlp_deeper_relu = dict(layers=[64, 64, 64], act_fun=tf.nn.relu)
mlp_shallower_relu = dict(layers=[32], act_fun=tf.nn.relu)

mlp_wider_tanh = dict(layers=[128, 128], act_fun=tf.nn.tanh)
mlp_default_tanh = dict(layers=[64, 64], act_fun=tf.nn.tanh)
mlp_deeper_tanh = dict(layers=[64, 64, 64], act_fun=tf.nn.tanh)
mlp_shallower_tanh = dict(layers=[32], act_fun=tf.nn.tanh)

mlp_wider_leaky_relu = dict(layers=[128, 128], act_fun=tf.nn.leaky_relu)
mlp_default_leaky_relu = dict(layers=[64, 64], act_fun=tf.nn.leaky_relu)
mlp_deeper_leaky_relu = dict(layers=[64, 64, 64], act_fun=tf.nn.leaky_relu)
mlp_shallower_leaky_relu = dict(layers=[32], act_fun=tf.nn.leaky_relu)

# defining individual strategies (models)
a2c_strategies = {
    # RELU
    'A2C_MLP_default_relu_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_default_relu),
        'A2C_MLP_default_relu_TL140',
        25_000,
    ),
    'A2C_MLP_wider_relu_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_wider_relu),
        'A2C_MLP_wider_relu_TL140',
        25_000,
    ),
    'A2C_MLP_deeper_relu_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_deeper_relu),
        'A2C_MLP_deeper_relu_TL140',
        25_000,
    ),
    'A2C_MLP_shallower_relu_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_shallower_relu),
        'A2C_MLP_shallower_relu_TL140',
        25_000,
    ),
    # TANH
    'A2C_MLP_default_tanh_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_default_tanh),
        'A2C_MLP_default_tanh_TL140',
        25_000,
    ),
    'A2C_MLP_wider_tanh_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_wider_tanh),
        'A2C_MLP_wider_tanh_TL140',
        25_000,
    ),
    'A2C_MLP_deeper_tanh_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_deeper_tanh),
        'A2C_MLP_deeper_tanh_TL140',
        25_000,
    ),
    'A2C_MLP_shallower_tanh_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_shallower_tanh),
        'A2C_MLP_shallower_tanh_TL140',
        25_000,
    ),
    # LEAKY RELU
    'A2C_MLP_default_leaky_relu_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_default_leaky_relu),
        'A2C_MLP_default_leaky_relu_TL140',
        25_000,
    ),
    'A2C_MLP_wider_leaky_relu_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_wider_leaky_relu),
        'A2C_MLP_wider_leaky_relu_TL140',
        25_000,
    ),
    'A2C_MLP_deeper_leaky_relu_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_deeper_leaky_relu),
        'A2C_MLP_deeper_leaky_relu_TL140',
        25_000,
    ),
    'A2C_MLP_shallower_leaky_relu_TL140': Strategy(
        partial(train_A2C, policy_kwargs=mlp_shallower_leaky_relu),
        'A2C_MLP_shallower_leaky_relu_TL140',
        25_000,
    ),
}

ddpg_strategies = {
    # RELU
    'DDPG_MLP_default_relu_TL140': Strategy(
       partial(train_DDPG, policy_kwargs=mlp_default_relu),
       'DDPG_MLP_default_relu_TL140',
       25_000,
    ),
    'DDPG_MLP_wider_relu_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_wider_relu),
        'DDPG_MLP_wider_relu_TL140',
        25_000,
    ),
    'DDPG_MLP_deeper_relu_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_deeper_relu),
        'DDPG_MLP_deeper_relu_TL140',
        25_000,
    ),
    'DDPG_MLP_shallower_relu_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_shallower_relu),
        'DDPG_MLP_shallower_relu_TL140',
        25_000,
    ),
    # TANH
    'DDPG_MLP_default_tanh_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_default_tanh),
        'DDPG_MLP_default_tanh_TL140',
        25_000,
    ),
    'DDPG_MLP_wider_tanh_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_wider_tanh),
        'DDPG_MLP_wider_tanh_TL140',
        25_000,
    ),
    'DDPG_MLP_deeper_tanh_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_deeper_tanh),
        'DDPG_MLP_deeper_tanh_TL140',
        25_000,
    ),
    'DDPG_MLP_shallower_tanh_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_shallower_tanh),
        'DDPG_MLP_shallower_tanh_TL140',
        25_000,
    ),
    # LEAKY RELU
    'DDPG_MLP_default_leaky_relu_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_default_leaky_relu),
        'DDPG_MLP_default_leaky_relu_TL140',
        25_000,
    ),
    'DDPG_MLP_wider_leaky_relu_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_wider_leaky_relu),
        'DDPG_MLP_wider_leaky_relu_TL140',
        25_000,
    ),
    'DDPG_MLP_deeper_leaky_relu_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_deeper_leaky_relu),
        'DDPG_MLP_deeper_leaky_relu_TL140',
        25_000,
    ),
    'DDPG_MLP_shallower_leaky_relu_TL140': Strategy(
        partial(train_DDPG, policy_kwargs=mlp_shallower_leaky_relu),
        'DDPG_MLP_shallower_leaky_relu_TL140',
        25_000,
    ),
}

ppo_strategies = {
    # RELU
    'PPO_MLP_default_relu_TL140': Strategy(
       partial(train_PPO, policy_kwargs=mlp_default_relu),
       'PPO_MLP_default_relu_TL140',
       25_000,
    ),
    'PPO_MLP_wider_relu_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_wider_relu),
        'PPO_MLP_wider_relu_TL140',
        25_000,
    ),
    'PPO_MLP_deeper_relu_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_deeper_relu),
        'PPO_MLP_deeper_relu_TL140',
        25_000,
    ),
    'PPO_MLP_shallower_relu_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_shallower_relu),
        'PPO_MLP_shallower_relu_TL140',
        25_000,
    ),
    # TANH
    'PPO_MLP_default_tanh_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_default_tanh),
        'PPO_MLP_default_tanh_TL140',
        25_000,
    ),
    'PPO_MLP_wider_tanh_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_wider_tanh),
        'PPO_MLP_wider_tanh_TL140',
        25_000,
    ),
    'PPO_MLP_deeper_tanh_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_deeper_tanh),
        'PPO_MLP_deeper_tanh_TL140',
        25_000,
    ),
    'PPO_MLP_shallower_tanh_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_shallower_tanh),
        'PPO_MLP_shallower_tanh_TL140',
        25_000,
    ),
    # LEAKY RELU
    'PPO_MLP_default_leaky_relu_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_default_leaky_relu),
        'PPO_MLP_default_leaky_relu_TL140',
        25_000,
    ),
    'PPO_MLP_wider_leaky_relu_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_wider_leaky_relu),
        'PPO_MLP_wider_leaky_relu_TL140',
        25_000,
    ),
    'PPO_MLP_deeper_leaky_relu_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_deeper_leaky_relu),
        'PPO_MLP_deeper_leaky_relu_TL140',
        25_000,
    ),
    'PPO_MLP_shallower_leaky_relu_TL140': Strategy(
        partial(train_PPO, policy_kwargs=mlp_shallower_leaky_relu),
        'PPO_MLP_shallower_leaky_relu_TL140',
        25_000,
    ),
}

strategies_dict = {
    **a2c_strategies,
    **ddpg_strategies,
    **ppo_strategies,
    'PPO_MLP_default_relu_TL140_TS3K': Strategy(
       partial(train_PPO, policy_kwargs=mlp_default_relu),
       'PPO_MLP_default_relu_TL140_TS3K',
       3_000,
    ),
    'PPO_MLP_default_relu_TL140_TS10K': Strategy(
       partial(train_PPO, policy_kwargs=mlp_default_relu),
       'PPO_MLP_default_relu_TL140_TS10K',
       10_000,
    ),
    'PPO_MLP_default_relu_TL140_TS20K': Strategy(
       partial(train_PPO, policy_kwargs=mlp_default_relu),
       'PPO_MLP_default_relu_TL140_TS20K',
       20_000,
    ),
    'PPO_MLP_default_relu_TL140_TS40K': Strategy(
       partial(train_PPO, policy_kwargs=mlp_default_relu),
       'PPO_MLP_default_relu_TL140_TS40K',
       40_000,
    ),
    'test': Strategy(
        partial(train_A2C, policy_kwargs=mlp_default_relu),
        'test',
        25_000,
    ),
    'DDPG_MLP_default_relu_TL1000': Strategy(
       partial(train_DDPG, policy_kwargs=mlp_default_relu),
       'DDPG_MLP_default_relu_TL1000',
       3_000,
    ),
    'A2C_defaultMLP_turbulence_level_140': Strategy(
        train_A2C,
        'A2C_defaultMLP_turbulence_level_140',
        25_000,
    ),
    'DDPG_defaultMLP_turbulence_level_140': Strategy(
        train_DDPG,
        'DDPG_defaultMLP_turbulence_level_140',
        10_000,
    ),
    'PPO_defaultMLP_turbulence_level_140': Strategy(
        train_PPO,
        'PPO_defaultMLP_turbulence_level_140',
        50_000,
    )
}
