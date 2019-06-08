import numpy as np

from gym import logger
from gym.vector.vector_env import VectorEnv
from gym.vector.utils import concatenate, create_empty_array

__all__ = ['SyncVectorEnv']


class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.

    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.

    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.

    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.

    episodic : bool (default: `False`)
        If `True`, then the environments run for a single episode (until
        `done=True`), and subsequent calls to `step` have an unexpected
        behaviour. If `False`, then the environments call `reset` at the end of
        each episode.
    """
    def __init__(self, env_fns, observation_space=None, action_space=None,
                 copy=True, episodic=False):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.episodic = episodic

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super(SyncVectorEnv, self).__init__(num_envs=len(env_fns),
            observation_space=observation_space, action_space=action_space)

        self._check_observation_spaces()
        self.observations = create_empty_array(self.single_observation_space,
            n=self.num_envs, fn=np.zeros)
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self._zero_observation = create_empty_array(self.single_observation_space,
            n=None, fn=np.zeros)

    def seed(self, seeds=None):
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def reset(self):
        self._dones[:] = False
        observations = []
        for i in range(self.num_envs):
            observation = self.envs[i].reset()
            observations.append(observation)
        concatenate(observations, self.observations, self.single_observation_space)

        return np.copy(self.observations) if self.copy else self.observations

    def step(self, actions):
        observations, infos = [], []
        for i, action in enumerate(actions):

            if self.episodic and self._dones[i]:
                observations.append(self._zero_observation)
                self._rewards[i] = 0.
                infos.append({})
                continue

            observation, self._rewards[i], self._dones[i], info = self.envs[i].step(action)
            if self._dones[i]:
                if self.episodic:
                    observation = self._zero_observation
                    info.update({'SyncVectorEnv.end_episode': True})
                else:
                    observation = self.envs[i].reset()

            observations.append(observation)
            infos.append(info)
        concatenate(observations, self.observations, self.single_observation_space)

        return (np.copy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards), np.copy(self._dones), infos)

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()

        for env in self.envs:
            env.close()

        self.closed = True

    def _check_observation_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                break
        else:
            return True
        raise RuntimeError('Some environments have an observation space '
            'different from `{0}`. In order to batch observations, the '
            'observation spaces from all environments must be '
            'equal.'.format(self.single_observation_space))
