# Advanced Machine Learning Challenge 1 - Report Summary

**Author:** Luis Fernando Palacios Flores  
**Course:** Advanced Topics in Machine Learning - Fall 2024, UniTS  
**Challenge:** Kernel Methods / Deep Learning Pipeline for FashionMNIST Dataset

## Overview

This report presents a comprehensive analysis of the FashionMNIST dataset using a hybrid pipeline that combines unsupervised dimensionality reduction techniques with supervised classification methods. The study explores the effectiveness of various kernel methods and deep learning approaches for fashion item classification.

## Dataset

- **Dataset:** FashionMNIST - A challenging replacement for the classic MNIST dataset
- **Characteristics:** Grayscale images of fashion items (28x28 pixels)
- **Classes:** 10 fashion categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Size:** 60,000 training samples, 10,000 test samples
- **Preprocessing:** Normalization using training set statistics, random shuffling with fixed seed (42)

## Section 1: Understanding Data Geometry

### 1.1 Linear PCA Analysis
- **Method:** Principal Component Analysis on flattened 784-dimensional feature vectors
- **Visualization:** 2D and 3D plots of first principal components
- **Findings:** Limited class separation due to linear nature of PCA
- **Explained Variance:** Analysis showed the cumulative variance explained by components

### 1.2 Kernel PCA with Gaussian (RBF) Kernel
- **Method:** Kernel PCA with Gaussian/RBF kernel
- **Hyperparameter Tuning:** Multiple gamma values tested (0.001, 0.01, 0.1, 1, 10)
- **Best Performance:** γ = 0.001 provided optimal class separation
- **Memory Optimization:** Used data subsampling to handle computational constraints
- **Results:** Significantly improved class separation compared to linear PCA

### 1.3 Kernel PCA with Polynomial Kernel
- **Method:** Kernel PCA with polynomial kernel
- **Hyperparameter Tuning:** Different polynomial degrees tested (2, 3, 4)
- **Performance:** Moderate improvement over linear PCA but inferior to Gaussian kernel
- **Computational Complexity:** Higher degrees increased computational burden

## Section 2: Bridging Unsupervised and Supervised Learning

### 2.1 Dimensionality Reduction Selection
- **Chosen Method:** Gaussian Kernel PCA (γ = 0.001) based on superior class separation
- **Components Used:** First 10 principal components for clustering analysis

### 2.2 Clustering Analysis
- **Methods Tested:**
  - K-Means clustering
  - Agglomerative clustering
- **Additional Technique:** t-SNE for non-linear dimensionality reduction
- **Evaluation Metrics:**
  - Adjusted Rand Index (ARI)
  - Silhouette Score
  - Clustering Quality Score

### 2.3 Best Clustering Results
Based on the clustering results CSV:
- **Top Performer:** t-SNE 3D + K-Means
  - ARI: 0.497 (highest among meaningful clusterings)
  - Silhouette Score: 0.369
  - Best balance between clustering quality and class separation

### 2.4 Manual Label Assignment
- **Method:** Visual inspection of cluster representatives
- **Process:** Expert labeling to map cluster IDs to fashion categories
- **Challenge:** Demonstrates the complexity of manual annotation in real-world scenarios

## Section 3: Supervised Classification

### 3.1 Support Vector Machine (SVM)
- **Kernel:** Gaussian/RBF kernel with optimized hyperparameters
- **Training Data:** Images with cluster-assigned labels
- **Performance:** Evaluated using confusion matrices and classification metrics

### 3.2 Fully Connected Neural Network (FCN)
- **Architecture:** Multiple hidden layers with various configurations
- **Hyperparameters:** Different network depths and widths tested
- **Training:** Standard backpropagation with appropriate loss functions
- **Results:** Generated learning curves and performance metrics

### 3.3 Convolutional Neural Network (CNN)
- **Architecture:** Convolutional layers followed by fully connected layers
- **Design:** Leverages spatial structure of image data
- **Comparison:** Performance compared against FCN architecture
- **Advantages:** Better feature extraction for image classification tasks

## Section 4: Pipeline Evaluation

### 4.1 Test Set Performance
- **Evaluation:** All three classifiers tested on FashionMNIST test set
- **Metrics:** Accuracy, precision, recall, F1-score
- **Label Mapping:** Cluster labels mapped to true fashion categories

### 4.2 Performance Comparison
The hybrid pipeline performance was evaluated against true labels, showing:
- **CNN:** Best performance due to spatial feature extraction capabilities
- **SVM:** Good performance with appropriate kernel selection
- **FCN:** Moderate performance limited by fully connected architecture

## Section 5: Fully Supervised Approach

### 5.1 Baseline Comparison
- **Method:** Training all classifiers with true FashionMNIST labels
- **Purpose:** Establish performance ceiling for comparison
- **Results:** Significantly higher accuracy than hybrid pipeline

### 5.2 Performance Analysis
- **CNN with True Labels:** Achieved highest classification accuracy
- **Comparison:** Demonstrated the impact of label quality on final performance
- **Insights:** Highlighted the importance of good unsupervised preprocessing

## Key Findings and Conclusions

### 1. Dimensionality Reduction Insights
- **Gaussian Kernel PCA** with γ = 0.001 provided the best class separation
- **t-SNE** proved highly effective for visualization and clustering preparation
- **Linear PCA** showed limitations for complex, non-linear data structures

### 2. Clustering Performance
- **t-SNE + K-Means** combination achieved the best unsupervised labeling
- **Adjusted Rand Index** of 0.497 indicates moderate success in recovering true structure
- Manual labeling remains challenging and time-intensive

### 3. Classification Results
- **CNN architecture** consistently outperformed other approaches
- **Spatial structure preservation** crucial for image classification tasks
- **Kernel SVM** provided competitive results with proper hyperparameter tuning

### 4. Pipeline Effectiveness
- **Hybrid approach** achieved reasonable performance without true labels
- **Performance gap** between hybrid and fully supervised approaches highlights importance of label quality
- **Computational efficiency** considerations important for large-scale applications

## Technical Contributions

1. **Comprehensive comparison** of linear and kernel-based dimensionality reduction methods
2. **Systematic evaluation** of clustering algorithms for pseudo-label generation
3. **Multi-architecture comparison** of classification approaches
4. **Performance analysis** across different supervision levels
5. **Practical insights** into computational constraints and optimization strategies

## Experimental Setup

- **Reproducibility:** Fixed random seed (42) for consistent results
- **Computational Optimization:** Data subsampling for memory-intensive operations
- **Comprehensive Evaluation:** Multiple metrics and visualization techniques
- **Systematic Approach:** Structured pipeline for fair comparison

## Future Directions

1. **Advanced Architectures:** Exploration of modern deep learning architectures
2. **Semi-supervised Learning:** Integration of semi-supervised techniques
3. **Transfer Learning:** Utilization of pre-trained models
4. **Computational Optimization:** More efficient kernel methods implementation
5. **Ensemble Methods:** Combination of multiple classification approaches

---
