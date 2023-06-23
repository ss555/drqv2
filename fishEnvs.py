
#%%
import math
import time
import cv2
import numpy as np
import timeit
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
from collections import deque
from rlutils.cnn_utils import save_buffer
from rlutils.utils import rungekutta4
#%%
from rlutils.env_wrappers import ActionSpec,ObservationSpec,Observation
class Fish_u3a_alpha_stack(gym.Env):
    """
    Description:
        A robot fish-results swimming in the water

    Source:
        The environment is built based on the Jesús Sánchez-Rodríguez's model

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Position x
        1       Velocity x'
        2       alpha
        3       alpha'

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Rest still(nee rien faire)
		1	  Tail to the positive +Gamma = +alpha
        2     Tail to the negative -Gamma = -alpha

        Note:

    Reward: TODO
         Reward of 1 is awarded if the agent reached the target velocity Ut.
         Reward of -1 is awarded if the alpha is superior to A/L = 0.3
         Reward of 0 is awarded for custom_Mujoco_tasks situations

    Starting State: TODO: rescale, N timesteps
         alpha, x_dot (all) are assigned a uniform random value in [-0.05..0.05]
         # The starting velocity of the fish-results is always assigned to 0.

    Episode Termination:
         The velocity of fish-results is more than the goal_velocity, for
		 after N_steps=3000 timesteps

	Eq Simplifie:
	K_m * x'' = - a''*a - x'**2
    alpha'' + 2*eta * omega * alpha' + omega**2(alpha-Gamma) = 0

    """

    def __init__(self,
                 Gamma=60,
                 tau=0.02,
                 seed=0,
                 rwd_method=1,
                 EP_STEPS=800,
                 omega0=11.8,
                 eta=1.13 / 2,
                 startCondition='random',
                 targetU=1,
                 n_stack=4,
                 kPunish=0.1,
                 ):

        super(Fish_u3a_alpha_stack, self).__init__()
        ### Control parameters
        self.rwd_method = rwd_method
        self.Gamma = Gamma
        self.tau = tau
        self.kinematics_integrator = 'symplectic'
        self.action_max = np.deg2rad(90)
        self.targetU = targetU
        self.n_stack = n_stack
        self.kPunish = kPunish

        self.startCondition = startCondition
        self.seed(seed)
        self.viewer = None
        self.N_STEPS = EP_STEPS
        self.target_counter = 0
        self.step_current = 0
        self.start = timeit.default_timer()

        ### physical parameters
        ## Servo motor to fish-results
        # self.Omega = Omega
        # self.delta_phi = delta_phi
        # self.ratio_phi_alpha = 0.48
        # self.ratio_phic_phi = 0.85
        # self.ratio_non_linear = 1.6 ## rad-2

        ## Fish
        self.mass = 1
        self.length = 0.08  ## in m
        self.omega0 = omega0  # * self.tau  # ~ 10
        self.eta = eta  # ~ 0.5 conjugé, harmonique + exp
        self.Ka2 = 12.9e-3  # K_\alpha'' in N.rad-1.s2
        self.Ka1 = 39.9e-3  # K_\alpha'
        # self.Ct = 0.04  ## Ct = rho l**4 = 1e3*(0.08)**4 =0.041
        self.Ct = self.Ka2
        self.Cd = 0.254  ## from Jesus's PRL paper

        low = np.array([-0.1,  ## x, Umax = 0.1, 0.1*768*tau ~ 1.6
                        -np.deg2rad(-90),  ## alpha, in rad
                        ],
                       dtype=np.float32)

        high = np.array([100,
                         np.deg2rad(90),
                         ],
                        dtype=np.float32)
        low = np.tile(low, self.n_stack)
        high = np.tile(high, self.n_stack)
        # does it need float32?, maybe float16 is sufficient
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Initiation of state and intermedian variables
        self.state = None
        self.alpha = None
        self.alpha_dot = None
        self.x = None
        self.x_dot = None
        self.avgAlpha = None
        self.dList = []

        # Determination of reward
        if self.rwd_method == 1:
            self.reward = self.reward_1
        elif self.rwd_method == 2:
            self.reward = self.reward_2
        elif self.rwd_method == 3:
            self.reward = self.reward_3
        # elif self.rwd_method == 4:
        #     self.reward= self.reward_4
        # elif self.rwd_method == 5:
        #     self.reward= self.reward_5
        # elif self.rwd_method == 6:
        #     self.reward= self.reward_6
        # elif self.rwd_method == 7:
        #     self.reward= self.reward_7

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_1(self, done):
        if not done:
            reward = self.x_dot
        else:
            reward = 0
        return reward

    def reward_2(self, done):
        if not done:
            reward = - abs(self.x_dot - self.targetU)
        else:
            reward = 0
        return reward

    def reward_3(self, done):
        if not done:
            reward = - abs(self.x_dot - self.targetU) - self.kPunish * abs(self.avgAlpha)
        else:
            reward = 0
        return reward

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        tmp = np.array(self.state)
        alpha = tmp[-1]
        alpha_dot = (tmp[-1] - tmp[-3]) / self.tau
        x = tmp[-2]
        x_dot = (tmp[-2] - tmp[-4]) / self.tau

        if action == 0:
            alpha_c = np.deg2rad(0)  ## degree2rad
        elif action == 1:
            alpha_c = np.deg2rad(self.Gamma)
        elif action == 2:
            alpha_c = np.deg2rad(-self.Gamma)
        else:
            print('invalid action chosen')
        # angular_diff = angular_diff * self.forceScale
        # K_m * x'' = - a''*a - x'**2
        # alpha'' + 2*eta * omega * alpha' + omega**2(alpha-Gamma) = 0

        # alpha_dd = self.forceScale * angular_force
        # alpha_dd = -alpha*action
        alpha_dd = -2 * self.eta * self.omega0 * alpha_dot - self.omega0 ** 2 * (alpha - alpha_c)
        x_dd = (- self.Cd * x_dot * np.abs(x_dot) - self.Ct * alpha_dd * alpha) / self.mass
        # x_dd = (- x_dot * np.abs(x_dot) - self.Ct * alpha_dd * alpha - self.K3 * alpha * alpha * x_dot * np.abs(x_dot))/self.K1

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * x_dd
            # alpha = alpha + self.tau * alpha_dot
            # alpha_dot = alpha_dot + self.tau * alpha_dd
        # elif self.kinematics_integrator == 'verlet':
        #     x = 2*x
        #     x = x + self.tau * x_dot
        #     alpha_dot = alpha_dot + self.tau * alpha_dd
        #     alpha = alpha + self.tau * alpha_dot
        elif self.kinematics_integrator == 'symplectic':
            x_dot = x_dot + self.tau * x_dd
            x = x + self.tau * x_dot
            alpha_dot = alpha_dot + alpha_dd * self.tau
            alpha = alpha + self.tau * alpha_dot
            # alpha_dot = alpha_dot + self.tau * alpha_dd
            # alpha = alpha + self.tau * alpha_dot

        self.state.extend(np.array([x, alpha], dtype=np.float32))
        self.x = x
        self.x_dot = x_dot
        self.alpha = alpha
        self.alpha_dot = alpha_dot

        done = bool(
            # self.target_counter >= self.target_number
            self.step_current > self.N_STEPS
        )
        reward = self.reward(done)

        self.step_current += 1
        self.avgAlpha = (self.avgAlpha * (self.step_current) + alpha) / (self.step_current + 1)

        dData = {'timestep': self.step_current,
                 'x': x,
                 'x_dot': x_dot,
                 'alpha': alpha,
                 'alpha_dot': alpha_dot,
                 'alpha_c': alpha_c,
                 'action': action}
        self.dList.append(dData)

        return np.array(self.state), reward, done, {}

    def reset(self):
        if self.startCondition == 'random':
            # self.alpha = self.np_random.uniform(low=-np.deg2rad(self.Gamma), high=np.deg2rad(self.Gamma))
            self.alpha = self.np_random.uniform(low=-0.05, high=0.05)
        else:
            self.alpha = 0
        self.state = deque(np.append(np.repeat(np.zeros(2), 3), np.array([0, self.alpha])), maxlen=self.n_stack * 2)

        self.steps_beyond_done = None
        self.step_current = 0
        self.x = 0
        self.x_dot = 0
        self.alpha_dot = 0
        self.avgAlpha = 0
        self.dList = [{'timestep': 0,
                       'x': 0,
                       'x_dot': 0,
                       'alpha': self.alpha,
                       'alpha_dot': 0,
                       'alpha_c': 0,
                       'action': 0}]
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, title="title"):
        self.df = pd.DataFrame.from_dict(self.dList)
        self.df.to_csv(title + '.csv')

class Fish_ammorti(gym.Env):

    def __init__(self,
                 Gamma=60,
                 tau=0.02,
                 seed=0,
                 rwd_method=1,
                 EP_STEPS=768,
                 omega0=11.8,
                 eta=1.13 / 2,
                 startCondition='random',
                 targetU=1,
                 kPunish=0.1,
                 discrete_actions =True
                 ):

        super(Fish_ammorti, self).__init__()
        ### Control parameters
        self.rwd_method = rwd_method
        self.Gamma = np.deg2rad(Gamma)
        self.tau = tau
        self.kinematics_integrator = 'symplectic'
        self.action_max = np.deg2rad(90)
        self.targetU = targetU
        self.kPunish = kPunish

        self.startCondition = startCondition
        self.seed(seed)
        self.viewer = None
        self.N_STEPS = EP_STEPS
        self.target_counter = 0
        self.step_current = 0
        self.start = timeit.default_timer()
        self.discrete_actions = discrete_actions

        ## Fish
        self.mass = 1
        self.length = 0.08  ## in m
        self.omega0 = omega0  # * self.tau  # ~ 10
        self.eta = eta  # ~ 0.5 conjugé, harmonique + exp
        self.Ka2 = 12.9e-3  # K_\alpha'' in N.rad-1.s2
        self.Ka1 = 39.9e-3  # K_\alpha'
        self.Ct = self.Ka2
        self.Cd = 0.254  ## from Jesus's PRL paper

        low = np.array([-0.1,  ## x, Umax = 0.1, 0.1*768*tau ~ 1.6
                        -100,
                        -np.deg2rad(75),  ## alpha, in rad
                        -100,
                        ],
                       dtype=np.float32)
        self.x_threshold = 50
        high = np.array([self.x_threshold,
                         100,
                         np.deg2rad(75),
                         100,
                         ],
                        dtype=np.float32)
        if self.discrete_actions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(-1,1,shape=(1,),dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Initiation of state and intermedian variables
        self.state = None
        self.avgAlpha = None
        self.dList = []
        self.viewer = None
        # Determination of reward
        self.reward = self.rew_map(counter=rwd_method)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def rew_map(self,counter):
        def reward_1(done):
            if not done:
                reward = self.state[1]
            else:
                reward = 0
            return reward

        def reward_2(done):
            if not done:
                reward = - abs(self.state[1] - self.targetU)
            else:
                reward = 0
            return reward

        def reward_3(done):
            if not done:
                reward = - abs(self.state[1] - self.targetU) - self.kPunish * abs(self.avgAlpha)
            else:
                reward = 0
            return reward
        if counter==1:
            return reward_1
        elif counter==2:
            return reward_2
        elif counter==3:
            return reward_3
        else:
            raise EnvironmentError
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        assert self.observation_space.contains(self.state), f'state {self.state}'
        [x, x_dot, alpha, alpha_dot] = self.state

        if self.discrete_actions:
            alpha_c = (action-1)*(self.Gamma)
        else:
            alpha_c = action[0] * (self.Gamma)

        # if alpha >= self.Gamma and alpha_c>=0:
        #     alpha_dd = 0
        #     alpha_dot=0
        # elif alpha<=-self.Gamma and alpha_c<=0:
        #     alpha_dd = 0
        #     alpha_dot=0
        alpha_dd = -2 * self.eta * self.omega0 * alpha_dot - self.omega0 ** 2 * (alpha - alpha_c)
        x_dd = (- self.Cd * x_dot * np.abs(x_dot) - self.Ct * alpha_dd * alpha) / self.mass

        x_dot = x_dot + self.tau * x_dd
        x = x + self.tau * x_dot
        alpha_dot = alpha_dot + alpha_dd * self.tau
        alpha = alpha + self.tau * alpha_dot

        self.state = [x, x_dot, alpha, alpha_dot]

        done = bool(
            self.step_current >= self.N_STEPS
        )
        reward = self.reward(done)

        self.step_current += 1
        self.avgAlpha = (self.avgAlpha * (self.step_current) + alpha) / (self.step_current + 1)

        return np.array(self.state), reward, done, {'ep_length':self.step_current}

    def reset(self):
        # print('reset fish-results env')
            #self.alpha = self.np_random.uniform(low=-0.05, high=0.05)
        self.state = np.zeros(shape=(4,),dtype=np.float32)
        self.avgAlpha = self.state[2]
        self.step_current = 0
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


