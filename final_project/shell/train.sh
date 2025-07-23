#!/bin/bash

# Deep Learning Training Script
# Usage: ./shell/train.sh [model]
# Models: mlp, cnn
# e.g. ./shell/train.sh mlp

set -e

# Get model argument
MODEL=${1:-"mlp"}

# Validate model
if [[ ! "$MODEL" =~ ^(mlp|cnn)$ ]]; then
    echo "❌ Error: Invalid model '$MODEL'. Use: mlp, cnn"
    exit 1
fi

# Start training
cd scripts
python main.py --model "$MODEL"
