#!/bin/bash
#SBATCH -A psl@v100
#SBATCH --job-name=ClimSim_training          # nom du job
#SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=40           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=10:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=ClimSim_training.out      # nom du fichier de sortie
#SBATCH --error=ClimSim_training.err       # nom du fichier d'erreur (ici commun avec la sortie)
##SBATCH --qos=qos_gpu-dev
 
module purge
module load tensorflow-gpu/py3/2.6.0
 
set -x

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY


srun python3 -u training.py
