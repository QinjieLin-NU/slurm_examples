#!/bin/bash
#SBATCH --job-name=hello_world
#SBATCH --output=hello_world.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# module load python/3.8
srun python helloworld.py

