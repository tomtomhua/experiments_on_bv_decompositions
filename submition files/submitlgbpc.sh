#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xhua24@wisc.edu
#SBATCH -D /workspace/xhua24/pmlbcode/PC
#SBATCH -J PMLBPC2
#SBATCH -t 71:59:00 # 24h
#SBATCH -p long
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=1024M
#SBATCH -o slurm_outputfile.out # will save stdout to `-D` directory
#SBATCH -e slurm_error_lgb_pc.out


module load python/miniconda
/workspace/xhua24/miniconda3/bin/python lgbParcom.py



