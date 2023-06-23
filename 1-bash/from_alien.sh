#rsync -av -e ssh alien:/home/install/anaconda3/envs/torch-back /home/sardor/anaconda3/envs/torchback
#rsync -av -e ssh alien:/home/install/Project/deepFish/servo-experiment/logs/filename.zip ./temp/ilename.zip
rsync -av -e ssh alien:/home/install/Project/FishCompute/VAE/logs ./VAE/

#rsync -av -e ssh alien:/home/install/Project/deepFish/servo-experiment/logs/filename.zip ./temp/linear_imitation_shot.zip
#rsync -av --delete -e ssh ./optuna/* alien:/home/install/Project/deepFish/optuna
#rsync -av --delete -e ssh ./deep_rl_utils/* alien:/home/install/Project/deepFish/deep_rl_utils
#rsync -av -e ssh ./* alien:/home/install/Project/deepFish/optuna
#rsync -av --delete -e ssh ./* i3s:/home/Sardor.Israilov/deepFish
