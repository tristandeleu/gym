import pytest
import numpy as np

from multiprocessing import TimeoutError
from gym.spaces import Box
from gym.error import (AlreadyPendingCallError, NoAsyncCallError,
                       ClosedEnvironmentError)
from gym.vector.tests.utils import make_env, make_slow_env

from gym.vector.async_vector_env import AsyncVectorEnv

@pytest.mark.parametrize('shared_memory', [True, False])
def test_create_async_vector_env(shared_memory):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    finally:
        env.close()

    assert env.num_envs == 8


@pytest.mark.parametrize('shared_memory', [True, False])
def test_reset_async_vector_env(shared_memory):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        observations = env.reset()
    finally:
        env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape


@pytest.mark.parametrize('shared_memory', [True, False])
def test_step_async_vector_env(shared_memory):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        observations = env.reset()
        actions = [env.single_action_space.sample() for _ in range(8)]
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


@pytest.mark.parametrize('shared_memory', [True, False])
def test_call_async_vector_env(shared_memory):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(4)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        observations = env.reset()
        images = env.call('render', mode='rgb_array')
        use_shaped_reward = env.call('use_shaped_reward')
    finally:
        env.close()

    assert isinstance(images, list)
    assert len(images) == 4
    for i in range(4):
        assert isinstance(images[i], np.ndarray)
        assert np.all(images[i] == observations[i])

    assert isinstance(use_shaped_reward, list)
    assert len(use_shaped_reward) == 4
    for i in range(4):
        assert isinstance(use_shaped_reward[i], bool)
        assert use_shaped_reward[i]


@pytest.mark.parametrize('shared_memory', [True, False])
def test_copy_async_vector_env(shared_memory):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory,
                             copy=True)
        observations = env.reset()
        observations[0] = 128
        assert not np.all(env.observations[0] == 128)
    finally:
        env.close()


@pytest.mark.parametrize('shared_memory', [True, False])
def test_no_copy_async_vector_env(shared_memory):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory,
                             copy=False)
        observations = env.reset()
        observations[0] = 128
        assert np.all(env.observations[0] == 128)
    finally:
        env.close()


@pytest.mark.parametrize('shared_memory', [True, False])
def test_reset_timeout_async_vector_env(shared_memory):
    env_fns = [make_slow_env(0.3, i) for i in range(4)]
    with pytest.raises(TimeoutError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            env.reset_async()
            observations = env.reset_wait(timeout=0.1)
        finally:
            env.close(terminate=True)


@pytest.mark.parametrize('shared_memory', [True, False])
def test_step_timeout_async_vector_env(shared_memory):
    env_fns = [make_slow_env(0., i) for i in range(4)]
    with pytest.raises(TimeoutError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            observations = env.reset()
            env.step_async([0.1, 0.1, 0.3, 0.1])
            observations, rewards, dones, _ = env.step_wait(timeout=0.1)
        finally:
            env.close(terminate=True)


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('shared_memory', [True, False])
def test_reset_out_of_order_async_vector_env(shared_memory):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(4)]
    with pytest.raises(NoAsyncCallError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            observations = env.reset_wait()
        except NoAsyncCallError as exception:
            assert exception.name == 'reset'
            raise
        finally:
            env.close(terminate=True)

    with pytest.raises(AlreadyPendingCallError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            actions = env.action_space.sample()
            observations = env.reset()
            env.step_async(actions)
            env.reset_async()
        except NoAsyncCallError as exception:
            assert exception.name == 'step'
            raise
        finally:
            env.close(terminate=True)


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('shared_memory', [True, False])
def test_step_out_of_order_async_vector_env(shared_memory):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(4)]
    with pytest.raises(NoAsyncCallError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            actions = env.action_space.sample()
            observations = env.reset()
            observations, rewards, dones, infos = env.step_wait()
        except AlreadyPendingCallError as exception:
            assert exception.name == 'step'
            raise
        finally:
            env.close(terminate=True)

    with pytest.raises(AlreadyPendingCallError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            actions = env.action_space.sample()
            env.reset_async()
            env.step_async(actions)
        except AlreadyPendingCallError as exception:
            assert exception.name == 'reset'
            raise
        finally:
            env.close(terminate=True)


@pytest.mark.parametrize('shared_memory', [True, False])
def test_already_closed_async_vector_env(shared_memory):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(4)]
    with pytest.raises(ClosedEnvironmentError):
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        env.close()
        observations = env.reset()


@pytest.mark.parametrize('shared_memory', [True, False])
def test_check_observations_async_vector_env(shared_memory):
    # CubeCrash-v0 - observation_space: Box(40, 32, 3)
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    # MemorizeDigits-v0 - observation_space: Box(24, 32, 3)
    env_fns[1] = make_env('MemorizeDigits-v0', 1)
    with pytest.raises(RuntimeError):
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        env.close(terminate=True)


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('shared_memory', [True, False])
def test_max_retries_async_vector_env(shared_memory):
    env_fns = [make_slow_env(0., i) for i in range(4)]
    with pytest.raises(ValueError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory,
                max_retries=2)
            env.reset()
            env.step([0.1, -1, 0.1, -1])
            env.step([-1, 0.1, 0.1, 0.1])
        finally:
            env.close(terminate=True)


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('shared_memory', [True, False])
def test_max_retries_observations_async_vector_env(shared_memory):
    env_fns = [make_slow_env(0., i) for i in range(4)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory,
            max_retries=2)
        env.reset()
        observations, rewards, dones, infos = env.step([0.1, -1, 0.1, -1])

        assert len(infos) == 4
        for j in [1, 3]:
            assert rewards[j] == 0.
            assert dones[j]
            assert 'AsyncVectorEnv.restart' in infos[j]
            assert infos[j]['AsyncVectorEnv.restart']

        observations, rewards, dones, infos = env.step([0.3, 0.1, 0.1, 0.1])
    finally:
        env.close(terminate=True)


@pytest.mark.parametrize('shared_memory', [True, False])
def test_episodic_async_vector_env(shared_memory):
    episode_lengths = [2, 5, 3, 1, 3]
    env_fns = [make_slow_env(0., i, length) for i, length
               in enumerate(episode_lengths)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory,
                            episodic=True)
        observations = env.reset()
        # Step 1
        actions = env.action_space.sample()
        observations, rewards, dones, infos = env.step(actions)
        assert dones[3]
        assert np.all(observations[3] == 0)
        assert 'AsyncVectorEnv.end_episode' in infos[3]
        assert infos[3]['AsyncVectorEnv.end_episode']
        assert not np.all(dones)
        assert np.any(observations[0] != 0)

        # Step 2
        actions = env.action_space.sample()
        observations, rewards, dones, infos = env.step(actions)
        for j in [0, 3]:
            assert dones[j]
            assert np.all(observations[j] == 0)
        assert 'AsyncVectorEnv.end_episode' in infos[0]
        assert infos[0]['AsyncVectorEnv.end_episode']
        assert not np.all(dones)
        assert np.any(observations[2] != 0)

        # Step 3
        actions = env.action_space.sample()
        observations, rewards, dones, infos = env.step(actions)
        for j in [0, 2, 3, 4]:
            assert dones[j]
            assert np.all(observations[j] == 0)
        assert 'AsyncVectorEnv.end_episode' in infos[2]
        assert infos[2]['AsyncVectorEnv.end_episode']
        assert not np.all(dones)
        assert np.any(observations[1] != 0)

        # Step 4
        actions = env.action_space.sample()
        observations, rewards, dones, infos = env.step(actions)
        for j in [0, 2, 3, 4]:
            assert dones[j]
            assert np.all(observations[j] == 0)
        assert 'AsyncVectorEnv.end_episode' not in infos[2]
        assert not np.all(dones)
        assert np.any(observations[1] != 0)

        # Step 5
        actions = env.action_space.sample()
        observations, rewards, dones, infos = env.step(actions)
        assert np.all(dones)
        assert np.all(observations == 0)
    finally:
        env.close()
