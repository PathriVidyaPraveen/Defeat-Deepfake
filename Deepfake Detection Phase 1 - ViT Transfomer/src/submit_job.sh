#!/bin/bash
#SBATCH --job-name=deepfake_vit      # Job name
#SBATCH --output=logs/out_%j.log     # Output log file
#SBATCH --error=logs/err_%j.log      # Error log file
#SBATCH --nodes=1                    # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Request 4 CPU cores
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=24:00:00              # Time limit (24 hours)
#SBATCH --partition=gpu              # Partition name (Check your cluster docs, might be 'gpu' or 'standard')

# 1. Load necessary modules (Adjust based on Param Seva documentation)
# module load cuda/11.8
# module load python/3.8

# 2. Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate your_env_name

# 3. Create logs directory if it doesn't exist
mkdir -p logs

# 4. Run the pipeline script
chmod +x run_pipeline.sh
./run_pipeline.sh