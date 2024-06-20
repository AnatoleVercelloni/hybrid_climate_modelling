#!/bin/bash
#SBATCH -A psl@cpu
#SBATCH --job-name=ClimSim_low_res_prepro          # nom du job
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --cpus-per-task=40           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:30:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=prepro_all.out      # nom du fichier de sortie
#SBATCH --error=prepro_all.err       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --qos=qos_cpu-dev
 
module purge
module load python

set -x

srun python3 -u prepro_all_data.py
date
