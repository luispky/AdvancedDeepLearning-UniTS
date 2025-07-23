# Advanced Machine Learning Challenge 2 - Report Summary

**Author:** Luis Fernando Palacios Flores  
**Subject:** Advanced Deep Learning - UniTS  
**Challenge:** Empirical Study on Function Learnability by Neural Networks

## Overview

This report presents an empirical investigation into the behavior of deep neural networks with respect to learning specific classes of functions under different training regimes. The study consists of two main parts examining different aspects of neural network learning dynamics.

## Part A: Under- and Over-parameterization in Teacher/Student Setup

### Experimental Setup

The first part investigates the effects of model capacity on learning through a teacher-student framework:

- **Teacher Model**: Fixed fully-connected network (100→75→50→10→1) with ReLU activations
- **Student Models**: Three variants with different capacities:
  - **Under-parameterized (Sᵤ)**: Single hidden layer (100→10→1)
  - **Equal-parameterized (Sₑ)**: Same as teacher (100→75→50→10→1)
  - **Over-parameterized (Sₒ)**: Four hidden layers (100→200→200→200→100→1)

### Training Parameters

- **Iterations**: 10,000 training iterations
- **Data**: 60,000 test samples from uniform distribution [0,2]¹⁰⁰
- **Initialization**: Standard normal distribution for weights and biases
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 128

### Part A Results

1. **Training Dynamics**: Generated comprehensive loss curves showing convergence behavior across different parameterization regimes
2. **Weight Distribution Analysis**: Compared final weight distributions between student and teacher models
3. **Generalization Performance**: Evaluated test set performance across all three student configurations
4. **Learning Rate Sensitivity**: Tested multiple learning rates (0.001, 0.01, 0.1) to optimize convergence

## Part B: Hierarchical vs Non-Hierarchical Function Learning

### Function Comparison Setup

The second part compares learning of functions with different structural properties:

- **Hierarchical Function (B₆)**: 6th-order multivariate complete Bell polynomial

```text
B₆(x₁,...,x₆) = x₁⁶ + 15x₂x₁⁴ + 20x₃x₁³ + 45x₂²x₁² + 15x₂³ + 60x₃x₂x₁ + 15x₄x₁² + 10x₃² + 15x₄x₂ + 6x₅x₁ + x₆
```

- **Non-Hierarchical Function (B̃₆)**: Scrambled version with same monomials but different variable assignments

### Model Architecture

- **Residual Network**: 9 layers (1 input + 8 hidden + 1 output)
- **Hidden Layers**: All size 50 with ReLU activations
- **Skip Connections**: ResNet-style connections between same-sized layers

### Training Configuration

- **Training Data**: 100,000 samples per function
- **Test Data**: 60,000 samples per function
- **Epochs**: 30 (with extended 500-epoch experiments)
- **Batch Size**: 20
- **Optimizer**: Adam with tuned learning rates
- **Input Range**: Uniform distribution [0,2]⁶

### Analysis Methods

1. **Training Dynamics**: Monitored training and test loss evolution
2. **Variable Sensitivity Analysis**: Conducted systematic input variable sweeps
3. **Aggregated Analysis**: Multiple trials (50 runs) for statistical robustness
4. **Comparative Study**: Direct comparison between hierarchical and non-hierarchical learning

### Part B Results

1. **Learning Efficiency**: Hierarchical functions showed different convergence patterns compared to non-hierarchical counterparts
2. **Generalization**: Systematic differences in test performance between B₆ and B̃₆
3. **Variable Dependencies**: Variable sweep analysis revealed how well networks captured individual input dependencies
4. **Robustness**: Multiple random seeds (11, 42, 1976, 1999, 2005) tested for statistical significance

## Technical Implementation

### Software Stack

- **Framework**: PyTorch for neural network implementation
- **Data Handling**: Custom data loaders and generators
- **Visualization**: Matplotlib and Seaborn for comprehensive plotting
- **Configuration**: YAML-based parameter management
- **Logging**: Structured logging for experiment tracking

### Experimental Control

- **Reproducibility**: Fixed random seeds across all experiments
- **Early Stopping**: Implemented to prevent overfitting
- **Model Persistence**: Systematic saving and loading of trained models
- **Results Tracking**: Comprehensive logging of all experimental parameters

## Key Insights

1. **Parameterization Effects**: Different capacity regimes led to distinct learning behaviors and final performance characteristics

2. **Hierarchical Structure Impact**: The inherent hierarchical structure of functions significantly influenced both learning dynamics and final model quality

3. **Generalization Patterns**: Systematic differences observed between learning hierarchical vs. non-hierarchical functions of equivalent complexity

4. **Training Stability**: Residual connections proved beneficial for learning complex polynomial functions

## Experimental Rigor

The study demonstrates high experimental standards with:

- Multiple random seed validation
- Comprehensive hyperparameter exploration
- Statistical aggregation across multiple trials
- Systematic comparison methodologies
- Detailed visualization of all results

## Conclusion

This empirical study provides valuable insights into how neural network architecture choices and function structure interact to influence learning outcomes. The teacher-student framework effectively isolated the effects of model capacity, while the hierarchical vs. non-hierarchical comparison revealed fundamental differences in how networks learn structured vs. unstructured functions of equivalent mathematical complexity.

The comprehensive experimental design, including proper statistical controls and multiple validation approaches, makes this a robust contribution to understanding neural network learning dynamics in controlled settings.
