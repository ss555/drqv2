rsync -av --exclude "" --delete -e ssh idris:/gpfswork/rech/yqs/ute89pr/drqv2_contrib/exp_local ./../drqv2_contrib-gen
#rsync -av --exclude -e ssh ~/Downloads/mujoco-2.1.1/* idris::/gpfswork/rech/yqs/ute89pr/.mujoco/*
#rsync -av -e ssh ~/Downloads/mujoco-2.1.1/* idris:/linkhome/rech/genzqh01/ute89pr/.mujoco/*
#ssh://ute89pr@idris:22/gpfswork/rech/yqs/ute89pr/.conda/envs/drqv2/bin/python3.8 -u /gpfswork/rech/yqs/ute89pr/drqv2_contrib/train.py