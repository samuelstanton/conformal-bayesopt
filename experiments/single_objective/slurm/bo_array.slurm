#!/bin/bash
#SBATCH --job-name=branin
#SBATCH --open-mode=append
#SBATCH --output=/home/wjm363/conformal-bayesopt/experiments/single_objective/slurm/outputs/%j_%x.out
#SBATCH --error=/home/wjm363/conformal-bayesopt/experiments/single_objective/slurm/errors/%j_%x.err
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH -c 4

#SBATCH --array=0-100

singularity exec --nv --overlay /scratch/wjm363/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "

source /ext3/env.sh
conda activate base

cd /home/wjm363/conformal-bayesopt/experiments/single_objective/

# python ./runner.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=50 \
# 	--problem=levy --dim=5 --batch_size=3 \
# 	--output=/scratch/wjm363/conformalbo/levy5_q3_${SLURM_ARRAY_TASK_ID}.pt
bash slurm/split_jobs.sh $SLURM_ARRAY_TASK_ID

"
