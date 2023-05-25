#!/bin/bash
#SBATCH --job-name=train_mnist
#SBATCH --output=train_mnist_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1


# Submit 5 jobs with different arguments
for i in {1..2}
do
  srun python train.py --lr 0.0${i} --epoch ${i} &
done
wait
