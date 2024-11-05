#!/bin/bash
#SBATCH -J mat_scaling       # Job name
#SBATCH -o mat_scaling.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e mat_scaling.o%j   # Name of stderr output file(%j expands to jobId)
#SBATCH -c 32            # Cores per task requested (1 task job)
# Needed 32 cores per A100 demanded
#SBATCH --mem-per-cpu=3G # memory per core demanded
#SBATCH --gres=gpu:a100   # Options for requesting 1GPU
#SBATCH -t 00:30:00      # Run time (hh:mm:ss) 
#SBATCH --partition=short

# Run the CUDA application
module load cesga/2020 cuda-samples/11.2
nvcc -o mat MatScaling.cu -lm -Xcompiler -Wall -Xcompiler -Wextra

# ---------------------#
echo "------4k x 4k executions------"
./mat 4000 4000 5 32
./mat 4000 4000 5 64
./mat 4000 4000 5 128

./mat 4000 4000 10 32
./mat 4000 4000 10 64
./mat 4000 4000 10 128
# ---------------------#
echo "------10k x 10k executions------"
./mat 10000 10000 5 32
./mat 10000 10000 5 64
./mat 10000 10000 5 128

./mat 10000 10000 10 32
./mat 10000 10000 10 64
./mat 10000 10000 10 128
# ---------------------#
echo "------20k x 20k executions------"
./mat 20000 20000 5 32
./mat 20000 20000 5 64
./mat 20000 20000 5 128

./mat 20000 20000 10 32
./mat 20000 20000 10 64
./mat 20000 20000 10 128
# ---------------------#
