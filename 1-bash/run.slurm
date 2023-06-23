#!/bin/bash
#SBATCH -A yqs@v100
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH -C v100-16g # reserver des GPU 16 Go seulement
#SBATCH --qos=qos_gpu-t3 # QoS
#SBATCH --output=TravailGPU%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=TravailGPU%j.err # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus MPI)
#SBATCH --gres=gpu:1 # reserver 4 GPU
#SBATCH --cpus-per-task=10 #server 10 CPU par tache (et memoire associee)
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --time=20:00:00

module purge # nettoyer les modules herites par defaut
module load pytorch-gpu/py3/1.11.0 # charger les modules
#srun python -u optuna/optuna_study_ppo_cnn_fish_ammorti.py #upright # executer son script
srun python -u optuna/optuna_study_ppo_vae_full_fish_ammorti.py #upright # executer son script
#srun python -u optuna/mujoco_optuna_study_ppo_fish_state.py #upright # executer son script