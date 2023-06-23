# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['HYDRA_FULL_ERROR'] = '1'

import pandas as pd

df = pd.read_csv('fish-results/medium/173641_/train.csv')
# df = pd.read_csv('./fish-results/easy/195923_/train.csv')
plt.plot(df['step'],df['episode_reward'])
# plt.show()
df = pd.read_csv('fish-results/medium/173641_/eval.csv')
plt.plot(df['step'],df['episode_reward'])
plt.legend(['train','eval'])
plt.show()
