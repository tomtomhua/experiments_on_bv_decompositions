#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xhua24@wisc.edu
#SBATCH -D /workspace/xhua24/pmlbcode/BV
#SBATCH -J PMLBBV0.2
#SBATCH -t 23:59:00 # 24h
#SBATCH -p short
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=1024M
#SBATCH -o slurm_outputfile.out # will save stdout to `-D` directory
#SBATCH -e slurm_error_bv_3.out


module load python/miniconda
/workspace/xhua24/miniconda3/bin/python BV3.py



