#!/bin/bash
#SBATCH --job-name=mffin
#SBATCH --ntasks=1                 # Correr una sola tarea
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=mffin_%j.out    # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=mffin_%j.err     # Output de errores (opcional)
#SBATCH --workdir=/home/samenabar/code/CLMAC/ReadGate   # Direccion donde correr el trabajo
#SBATCH --nodelist=grievous
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:Geforce-RTX:1
#SBATCH --partition=ialab-high
#SBATCH --mem=50GB

pwd; hostname; date

echo "Inicio entrenamiento Maffn"

source ~/.venvs/cuda10/bin/activate

python3 code/main.py --cfg cfg/.yml --logcomet > _$SLURM_JOBID.out 2>&1 &
wait

echo "Finished with job $SLURM_JOBID"
