#!/bin/bash
#SBATCH -A yqs@v100
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH -C v100-16g # reserver des GPU 16 Go seulement
#SBATCH --qos=qos_gpu-dev # QoS
#SBATCH --output=TravailGPU%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=TravailGPU%j.err # fichier d’erreur (%j = job ID)
#SBATCH --time=00:10:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus MPI)
#SBATCH --gres=gpu:1 # reserver 4 GPU
#SBATCH --cpus-per-task=1 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
module purge # nettoyer les modules herites par defaut
module load pytorch-gpu/py3/1.10.1 # charger les modules
conda activate
srun python -u train.py task=fish_upright # executer son script