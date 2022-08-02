#!/bin/bash
#SBATCH --job-name=tr_ucb_baseline
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ss13641/code/remote/conformal-bayesopt/experiments/single_objective/output/%x_%j.out
#SBATCH --error=/scratch/ss13641/code/remote/conformal-bayesopt/experiments/single_objective/error/%x_%j.err
#SBATCH --export=ALL
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-74

singularity exec --nv --overlay ${SCRATCH}/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "

source /ext3/env.sh
export PATH=/ext3/miniconda3/envs/conf-bo-env/bin:${PATH}
conda activate conf-bo-env

cd /home/ss13641/code/remote/conformal-bayesopt/experiments/single_objective/

bash ./slurm/split_jobs.sh $SLURM_ARRAY_TASK_ID $SLURM_JOB_NAME
"
