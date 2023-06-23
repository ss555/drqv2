'''
uses openFish.xml defines 2 classes Physics and Swim(A Fish `Task` for swimming with smooth reward.
simulation of a fish-results swimming in a tank with mujoco, uses dm_control
Use of stl files to create the fish-results body in a tank.
'''
import collections
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = 40
_CONTROL_TIMESTEP = .05
_JOINTS = ['dc_motor']
# SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('openFish.xml'), common.ASSETS

# @SUITE.add('benchmarking')
def swim(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Fish Swim task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Swim(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
      **environment_kwargs)

class Physics(mujoco.Physics):
    def torso_velocity(self):
        """Returns velocities and angular velocities of the torso."""
        return self.data.sensordata

    def joint_velocities(self):
        """Returns the joint velocities."""
        return self.named.data.qvel[_JOINTS]

    def joint_angles(self):
        """Returns the joint positions."""
        return self.named.data.qpos[_JOINTS]

    def distance_to_target(self, img=None):
        return mouth_to_target_global.dot(data.geom_xmat['mouth'].reshape(3, 3))

class Swim(base.Task):
  """A Fish `Task` for swimming with smooth reward."""

  def __init__(self, random=None):
    """Initializes an instance of `Swim`.
    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""

    quat = self.random.randn(4)
    physics.named.data.qpos['root'][3:7] = quat / np.linalg.norm(quat)
    for joint in _JOINTS:
      physics.named.data.qpos[joint] = self.random.uniform(-.2, .2)
    # Randomize target position.
    physics.named.model.geom_pos['target', 'x'] = self.random.uniform(-1000, -500)
    physics.named.model.geom_pos['target', 'y'] = self.random.uniform(-.4, .4)
    physics.named.model.geom_pos['target', 'z'] = self.random.uniform(.1, .3)
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of joints, target direction and velocities."""
    #overwrite dynamics eqs
    '''
    alpha_dd = -2 * self.eta * self.omega0 * alpha_dot - self.omega0 ** 2 * (alpha - alpha_c)
x_dd = (- self.Cd * x_dot * np.abs(x_dot) - self.Ct * alpha_dd * alpha) / self.mass
    '''
    # alpha,alpha_dot=
    physics.named.xmat['tail','zz']
    # physics.named.model.geom_pos[['body','fins'], 'x'] =
    obs = collections.OrderedDict()
    obs['joint_angles'] = physics.joint_angles()
    obs['upright'] = physics.upright()
    obs['target'] = physics.mouth_to_target()
    obs['velocity'] = physics.velocity()
    return obs

  # def mj_step(const mjModel *m, mjData *d): IN C
  #   print(mjModel)
  #   def udd_callback:


  def get_reward(self, physics):
    """Returns a smooth reward."""
    radii = physics.named.model.geom_size[['mouth', 'target'], 0].sum()
    in_target = rewards.tolerance(np.linalg.norm(physics.mouth_to_target()), bounds=(0, radii), margin=2*radii)
    return (7*in_target + is_upright) / 8

