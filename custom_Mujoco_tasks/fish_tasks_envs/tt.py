from dm_control import suite
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="fish", task_name="swim")
action_spec = env.action_spec()

# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)

# Launch the viewer application.
#, policy=random_policy)
# viewer.launch(env, policy=random_policy)
print('done')
with env.physics.reset_context():
  env.physics.named.data.geom_xpos=np.random.random(env.physics.named.data.geom_xpos.shape) # qpos['torso'] = np.pi

viewer.launch(env)