# Advanced Deep Learning at the University of Trieste, 2024–2025

## Overview

This repository contains my coursework for the Advanced Deep Learning course at the University of Trieste during the 2024–2025 academic year. The work spans theoretical foundations, empirical studies, and research in deep learning.

## Repository Structure

### 📚 Challenges
Two comprehensive assignments exploring fundamental concepts in machine learning and deep learning:

- **[Challenge 1](challenges/01/)**: Kernel Methods & Deep Learning Pipeline
  - **Focus**: Hybrid unsupervised-supervised learning pipeline for FashionMNIST
  - **Techniques**: Kernel PCA, clustering algorithms, SVM, FCN, CNN
  - **Key Insight**: Comparison of dimensionality reduction methods and their impact on classification performance

- **[Challenge 2](challenges/02/)**: Neural Network Function Learnability
  - **Focus**: Empirical study of neural network learning dynamics
  - **Techniques**: Teacher-student framework, hierarchical vs non-hierarchical functions
  - **Key Insight**: How model capacity and function structure affect learning outcomes

### 🎯 Final Project
A research project investigating advanced deep learning architectures:

- **[Group Equivariant CNNs & Low Coherence MLPs](final_project/)**
  - **G-CNNs**: Implementation of group-equivariant convolutional networks exploiting data symmetries
  - **Low Coherence MLPs**: Application of frame theory principles to neural network weight matrices
  - **Research Questions**: Does group equivariance improve performance on symmetric datasets? Can low-coherence frames enhance MLP training?

## Key Contributions

### Challenge 1: Kernel Methods Pipeline
- Comprehensive comparison of linear PCA vs kernel PCA (Gaussian, polynomial)
- Systematic evaluation of clustering algorithms for pseudo-label generation
- Multi-architecture classification comparison (SVM, FCN, CNN)
- Performance analysis across different supervision levels

### Challenge 2: Function Learnability
- Teacher-student framework with under/equal/over-parameterized models
- Hierarchical vs non-hierarchical function learning comparison
- Residual network analysis with systematic variable sensitivity studies
- Statistical robustness through multiple random seed validation

### Final Project: Advanced Architectures
- **G-CNN Implementation**: Group-equivariant convolutions for cyclic and dihedral groups
- **Frame Theory Application**: Low-coherence frame optimization for MLP weights
- **Comprehensive Evaluation**: Performance analysis on MNIST, Fashion-MNIST, CIFAR-10
- **Visualization Tools**: Kernel weight analysis and group transformation visualization

## Technical Stack

- **Frameworks**: PyTorch, PyTorch Lightning, TensorBoard
- **Data Processing**: Custom data loaders, dimensionality reduction techniques
- **Analysis**: Statistical evaluation, visualization, experiment tracking
- **Configuration**: YAML-based parameter management
- **Reproducibility**: Fixed random seeds, comprehensive logging

## Getting Started

Each subdirectory contains detailed documentation and setup instructions:

- **Challenges**: See individual `Report_Summary.md` files for detailed analysis
- **Final Project**: Comprehensive README with setup, usage, and results

## Results & Documentation

- **Challenge Reports**: Detailed PDF reports with comprehensive analysis
- **Project Slides**: Presentation materials available in the final project directory
- **Code Documentation**: Well-documented implementations with usage examples
- **Experimental Results**: Plots, logs, and analysis scripts for reproducibility

---

