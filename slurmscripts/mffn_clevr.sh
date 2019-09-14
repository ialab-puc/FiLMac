#!/bin/bash
#SBATCH --job-name=mffin_clevr
#SBATCH --ntasks=2                 # Correr una sola tarea
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=mffin_%j.out    # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=mffin_%j.err     # Output de errores (opcional)
#SBATCH --workdir=/home/samenabar/code/filmac/FiLMac   # Direccion donde correr el trabajo
#SBATCH --nodelist=grievous
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:Geforce-RTX:1
#SBATCH --partition=ialab-high
#SBATCH --mem=50GB

pwd; hostname; date

echo "Inicio entrenamiento Maffn"

source ~/.venvs/cuda10/bin/activate

# python3 code/main.py --cfg cfg/nb1_gqa.yml --logcomet > nb1_gqa_$SLURM_JOBID.out 2>&1 &
# python3 code/main.py --cfg cfg/nb0_gqa.yml --logcomet > nb0_gqa_$SLURM_JOBID.out 2>&1 &
python3 code/main.py --cfg cfg/nb1_clevr.yml --logcomet > nb1_clevr_$SLURM_JOBID.out 2>&1 &
python3 code/main.py --cfg cfg/nb0_clevr.yml --logcomet > nb0_clevr_$SLURM_JOBID.out 2>&1 &

wait

echo "Finished with job $SLURM_JOBID"
