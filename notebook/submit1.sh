#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xhua24@wisc.edu
#SBATCH -D /workspace/xhua24/pmlbcode
#SBATCH -J PMLB1
#SBATCH -t 12:00:00 # 24h
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=1024M
#SBATCH -o slurm_outputfile.out # will save stdout to `-D` directory
#SBATCH -e slurm_error.out


module load python/miniconda
/workspace/xhua24/pmlbcode
python BVin1st.py



