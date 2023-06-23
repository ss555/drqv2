import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['HYDRA_FULL_ERROR'] = '1'
from pathlib import Path
import hydra
import torch
import dmc

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train_dm_custom import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    env= dmc.make_dm_env(cfg.task_name, cfg.frame_stack, cfg.action_repeat, cfg.seed)
    env.reset()


if __name__ == '__main__':
    main()
