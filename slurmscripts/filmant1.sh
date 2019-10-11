#!/bin/bash
#SBATCH --job-name=filmant_clevr
#SBATCH --ntasks=2                 # Correr una sola tarea
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=filmant_%j.out    # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=filmant_%j.err     # Output de errores (opcional)
#SBATCH --workdir=/home/rmanterola/repos/FiLMac   # Direccion donde correr el trabajo
#SBATCH --nodelist=grievous
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:Geforce-RTX:1
#SBATCH --partition=ialab-high
#SBATCH --mem=50GB

pwd; hostname; date

echo "Inicio entrenamiento filmant"

source ~/virtualenv/film/bin/activate

python3 code/main.py --cfg cfg/filmant1.yml --logcomet > 3transformer_$SLURM_JOBID.out 2>&1 &
# python3 code/main.py --cfg cfg/filmant2.yml --logcomet > filmant2_clevr_$SLURM_JOBID.out 2>&1 &

wait

echo "Finished with job $SLURM_JOBID"
