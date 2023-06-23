import os.path

ROOT = os.getcwd()
import dm_env
from dm_control import suite
# from suite.openFish import swim

def stepActionEnv(env,action,print_obs = False,timesteps=1):
    state_array, rew_array = [], []
    for i in range(timesteps):
        obs, rew, done, info = env.step(action)
        state_array.append(env.unwrapped.state)
        rew_array.append(rew)
        env.render()
        if print_obs:
            print(obs)
        if done:
            return state_array, rew_array, done
    return state_array, rew_array, done

class Wrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def step(self, action):
        time_step = self._env.step(action)
# env = suite.load('fish-results', 'swim', task_kwargs={'random': 1}, visualize_reward=False)
# env = suite.load('dog', 'stand', task_kwargs={'random': 1}, visualize_reward=False)
env = suite.load('openFish', 'swim', task_kwargs={'random': 1}, visualize_reward=False)
# env = swim(time_limit=1e3)


env.step([1.0])
stepActionEnv(env=env,timesteps=20)