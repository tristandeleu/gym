import pytest
import numpy as np

from gym.spaces import Box
from gym.vector.tests.utils import make_env

from gym.vector.sync_vector_env import SyncVectorEnv

def test_create_sync_vector_env():
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = SyncVectorEnv(env_fns)
    finally:
        env.close()

    assert env.num_envs == 8


def test_reset_sync_vector_env():
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = SyncVectorEnv(env_fns)
        observations = env.reset()
    finally:
        env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape


@pytest.mark.parametrize('use_single_action_space', [True, False])
def test_step_sync_vector_env(use_single_action_space):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = SyncVectorEnv(env_fns)
        observations = env.reset()
        if use_single_action_space:
            actions = [env.single_action_space.sample() for _ in range(8)]
        else:
            actions = env.action_space.sample()
        observations, rewards, dones, _ = env.step(actions)
    finally:
        env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape

    assert isinstance(rewards, np.ndarray)
    assert isinstance(rewards[0], (float, np.floating))
    assert rewards.ndim == 1
    assert rewards.size == 8

    assert isinstance(dones, np.ndarray)
    assert dones.dtype == np.bool_
    assert dones.ndim == 1
    assert dones.size == 8


def test_call_sync_vector_env():
    env_fns = [make_env('CubeCrash-v0', i) for i in range(4)]
    try:
        env = SyncVectorEnv(env_fns)
        observations = env.reset()
        images = env.call('render', mode='rgb_array')
        use_shaped_reward = env.call('use_shaped_reward')
    finally:
        env.close()

    assert isinstance(images, tuple)
    assert len(images) == 4
    for i in range(4):
        assert isinstance(images[i], np.ndarray)
        assert np.all(images[i] == observations[i])

    assert isinstance(use_shaped_reward, tuple)
    assert len(use_shaped_reward) == 4
    for i in range(4):
        assert isinstance(use_shaped_reward[i], bool)
        assert use_shaped_reward[i]


def test_set_attr_sync_vector_env():
    env_fns = [make_env('CubeCrash-v0', i) for i in range(4)]
    try:
        env = SyncVectorEnv(env_fns)
        env.set_attr('use_shaped_reward', [True, False, False, True])
        use_shaped_reward = env.get_attr('use_shaped_reward')
        assert use_shaped_reward == (True, False, False, True)
    finally:
        env.close()


def test_check_observations_sync_vector_env():
    # CubeCrash-v0 - observation_space: Box(40, 32, 3)
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    # MemorizeDigits-v0 - observation_space: Box(24, 32, 3)
    env_fns[1] = make_env('MemorizeDigits-v0', 1)
    with pytest.raises(RuntimeError):
        env = SyncVectorEnv(env_fns)
        env.close()
