#rsync -av -e ssh /home/sardor/.mujoco alien:/home/install/
rsync -av -e ssh --exclude '.git/*' ./* alien:/home/install/Project/FishCompute/*
#rsync -av -e ssh --exclude '.git/*' ./* alien:/home/install/Project/FishCompute/*

#rsync -av --delete -e ssh ./VAE/* alien:/home/install/Project/FishCompute/VAE
#rsync -av -e ssh ./deep_rl_utils/* alien:/home/install/Project/FishCompute/deep_rl_utils

#rsync -av -e ssh /home/sardor/1-THESE/4-sample_code/00-current/drqv2_contrib alien:/home/install/Project/drqv2_contrib
#rsync -av -e ssh /media/sardor/b/00-current alien:/home/install/Project/00-current


#rsync -av -e ssh ./servo-experiment/npz_log/28/* alien:/home/install/Project/deepFish/servo-experiment/npz_log/25
#rsync -av -e ssh ./servo-experiment/npz_log/29/* alien:/home/install/Project/deepFish/servo-experiment/npz_log/26
#rsync -av --delete -e ssh ./optuna/* alien:/home/install/Project/deepFish/optuna
#rsync -av --delete -e ssh ./deep_rl_utils/* alien:/home/install/Project/deepFish/deep_rl_utils
#rsync -av -e ssh ./* alien:/home/install/Project/deepFish/optuna
#rsync -av --delete -e ssh ./* i3s:/home/Sardor.Israilov/deepFish