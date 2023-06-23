# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Fish Domain."""
import collections
import csv
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np
import os


_DEFAULT_TIME_LIMIT = 40
_CONTROL_TIMESTEP = .04
_JOINTS = ['root',
           'tail1',
           # 'tail_twist',
           'tail2']
           # 'finright_roll',
           # 'finright_pitch',
           # 'finleft_roll',
           # 'finleft_pitch']
SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  # return common.read_model(os.path.join(os.path.abspath('./'),'fishD.xml')), common.ASSETS
  return common.read_model(os.path.join(os.path.dirname(os.path.realpath(__file__)),'fishD.xml')), common.ASSETS
  # return common.read_model(sys.a'fishD.xml'), common.ASSETS


@SUITE.add('benchmarking')
def swim(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Fish Swim task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Swim(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Fish domain."""

  def upright(self):
    """Returns projection from z-axes of torso to the z-axes of worldbody."""
    return self.named.data.xmat['torso', 'zz']

  def torso_velocity(self):
    """Returns velocities and angular velocities of the torso."""
    return self.data.sensordata

  def joint_velocities(self):
    """Returns the joint velocities."""
    return self.named.data.qvel[_JOINTS]

  def joint_angles(self):
    """Returns the joint positions."""
    return self.named.data.qpos[_JOINTS]

  def mouth_to_target(self):
    """Returns a vector, from mouth to target in local coordinate of mouth."""
    data = self.named.data
    mouth_to_target_global = data.geom_xpos['target'] - data.geom_xpos['mouth']
    return mouth_to_target_global.dot(data.geom_xmat['mouth'].reshape(3, 3))




class Swim(base.Task):
  """A Fish `Task` for swimming with smooth reward."""

  def __init__(self, random=None,reward_mode='speed'):
    """Initializes an instance of `Swim`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self.reward_mode = reward_mode
    self.init=True
    f1 = open('./log-fish.csv', "w") #erase
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""

    # quat = self.random.randn(4)
    # physics.named.data.qpos['root'][3:7] = quat / np.linalg.norm(quat)
    # for joint in _JOINTS:
    #   physics.named.data.qpos[joint] = self.random.uniform(-.2, .2)
    # # Randomize target position.
    # physics.named.model.geom_pos['target', 'x'] = self.random.uniform(-.4, .4)
    # physics.named.model.geom_pos['target', 'y'] = self.random.uniform(-.4, .4)
    # physics.named.model.geom_pos['target', 'z'] = self.random.uniform(.1, .3)
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of joints, target direction and velocities."""
    obs = collections.OrderedDict()
    obs['joint_angles'] = physics.joint_angles()
    # obs['upright'] = physics.upright()
    # obs['target'] = physics.mouth_to_target()
    obs['velocity'] = physics.joint_velocities()
    return obs

  def get_reward(self, physics):
    """Returns a smooth reward."""
    radii = physics.named.model.geom_size[['mouth', 'target'], 0].sum()
    in_target = rewards.tolerance(np.linalg.norm(physics.mouth_to_target()), bounds=(0, radii), margin=2*radii)
    # is_upright = 0.5 * (physics.upright() + 1)
    if self.reward_mode=='speed':
      # return np.array([physics.named.data.xpos['torso', 'y']])#good
      # return physics.named.data.qpos['root']
      # t=physics.named.data.qvel['root']
      # print(physics.named.data.qvel['root'])
      return physics.named.data.qvel['root'][0]
    return in_target

  # def after_step(self,physics):
  #     row = {'j_p': physics.named.data.qpos[_JOINTS[0]][0], 'j_v': physics.named.data.qvel[_JOINTS[0]][0], 'j_acc': physics.named.data.qacc[_JOINTS[0]][0],
  #            's_acc': physics.named.data.qacc["root"][0], 's_p': physics.named.data.qpos["root"][0], 's_v': physics.named.data.qvel["root"][0], 'F_x':physics.named.data.sensordata[6],'F_y':physics.named.data.sensordata[7],'F_z':physics.named.data.sensordata[8]}
  #
  #     with open('./log-fish-results.csv', 'a', newline='') as f:
  #     # with open('./log.csv', 'a', newline='') as f:
  #         # create the csv writer
  #         writer = csv.DictWriter(f, fieldnames=row.keys())
  #         if self.init:
  #             writer.writeheader()
  #             self.init = False
  #         # write a row to the csv file
  #         writer.writerow(row)
  #         f.close()