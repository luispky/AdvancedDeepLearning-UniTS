#!/bin/bash
#SBATCH --job-name=teacher_student
#SBATCH --output=../logs/teacher_student.log  # Redirect stdout to logs directory
#SBATCH --error=../logs/teacher_student.err   # Redirect stderr to logs directory
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Ensure the logs directory exists
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"

# Load necessary modules and activate environment
module load cuda/12.1  # Load CUDA
source /scratch/lpalacio/miniconda/bin/activate DL

# Run the main script
python teacher_student_parallel.py