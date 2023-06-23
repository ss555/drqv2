'''
freq_scanning of action space in mode [sin, square, tri] for 1d speed maximisation of mujoco
'''
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['HYDRA_FULL_ERROR'] = '1'
import hydra
import dmc
import numpy as np
import pandas as pd
from rlutils.utils import gait_harmonic_sim
from rlutils.plot_utils import config_paper
from matplotlib import pyplot as plt

colors = config_paper()
max_freq=10
freqs = np.linspace(0.3, max_freq, 20)
phis = [1]#np.linspace(0, 2*np.pi, 20)

def compute_sim_steps(env, phi, freq, mode='sin',tau=0.01,include_obs=False,dmcontrol=False):
    acts = gait_harmonic_sim(phi=phi, omega=freq, mode=mode, dt=tau)
    dList=[]
    if include_obs:
        dList.append({'obs':[],'rew':[],'done':[]})#,'info':[]
    else:
        dList.append(0)
    env.reset()
    if include_obs:
        for a in acts:
            if dmcontrol:
                step = env.step([a])
                dList.append({'obs': step.obs, 'rew': step.rew, 'done': step.done})
            else:
                obs, rew, done, info = env.step([a])
                dList.append({'obs':obs,'rew':rew,'done':done})
    else:
        for a in acts:
            if dmcontrol:
                step = env.step([a])
                dList.append(step.reward)
            else:
                obs, rew, done, info = env.step([a])
                dList.append(rew)
    # return {'rew':np.array(dList).sum()}
    return dList

@hydra.main(config_path='../../cfgs', config_name='config')
def main(cfg):
    # env = dmc.make(cfg.task_name, cfg.frame_stack, cfg.action_repeat, cfg.seed)
    env = dmc.make_dm_env(cfg.task_name, cfg.frame_stack, cfg.action_repeat, cfg.seed)
    # with mp.Pool(12) as pool:
    #     dList = pool.starmap(compute_sim_steps, [(env, phi, freq, mode) for freq in freqs for phi in phis for mode in ['sin','square','tri']])
    dfList=[]
    dt=0.08
    for freq in freqs:
        for phi in phis:
            for mode in ['sin', 'square', 'tri']:
                rews = compute_sim_steps(env, phi, freq, mode, tau=dt, include_obs=False, dmcontrol=True)
                dfList.append({'freq':freq,'phi':phi,'mode':mode,'rews':rews,'mean_period_rew':np.mean(rews[-int(1000*dt*freq):])})
                # df.to_csv(f'./logs/mujoco_forcing_{mode}_{freq}_{phi}.csv', index=False)
    # save
    df = pd.DataFrame.from_dict(dfList)
    df.to_csv(f'./mujoco_forcing.csv', index=False)

    fig,ax = plt.subplots()
    ax.plot(freqs,dfList['mean_period_rew'])
    plt.legend()
    plt.savefig(f'./mujoco_forcing{max_freq}.png')
    plt.show()

    dfp = df.pivot(index='freq', columns='mode', values='mean_period_rew')
    dfp.plot(color=colors)
    plt.savefig('./mujoco_forcing_pivot.png')

if __name__ == '__main__':
    main()
