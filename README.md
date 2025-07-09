# Clustering Analysis and Feature Selection Project

This repository contains advanced clustering analysis and feature selection algorithms, continuing work from the [YueyOare/coursework](https://github.com/YueyOare/coursework) repository.

## Project Overview

This project implements comprehensive clustering analysis using multiple algorithms with hyperparameter optimization and automated feature selection. The main focus is on:

- **Multi-algorithm clustering optimization** using Optuna
- **Automated feature selection** through greedy elimination
- **Dimensionality reduction** with PCA and UMAP
- **Comprehensive evaluation** using multiple metrics

## Features

### Clustering Algorithms
- K-Means
- DBSCAN
- Gaussian Mixture Models
- Agglomerative Clustering
- Spectral Clustering
- BIRCH
- Mean Shift
- Affinity Propagation
- OPTICS
- HDBSCAN

### Dimensionality Reduction
- Principal Component Analysis (PCA)
- Uniform Manifold Approximation and Projection (UMAP)

### Evaluation Metrics
- Silhouette Score
- Calinski-Harabasz Index
- Likelihood (for Gaussian Mixture Models)

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── RDBA_BEKHTEREV2.xlsx          # Original dataset
│   ├── preprocessed/
│   │   ├── prepared_data.csv             # Preprocessed data with UMAP
│   │   ├── prepared_data_nonumap.csv     # Preprocessed data without UMAP
│   │   └── prepared_data.xlsx            # Excel format of preprocessed data
│   ├── preprocessing.ipynb               # Data preprocessing pipeline
│   ├── visualization_results.ipynb       # Results visualization and analysis
│   ├── clustering.py                     # Main clustering optimization script
│   ├── feature_selection.py              # Feature selection implementation
│   ├── clustering_results.csv            # Clustering results
│   ├── clustering_results_1.csv          # Extended clustering results
│   ├── cluster_means.xlsx                # Cluster centroids analysis
│   └── *.png                            # Generated visualizations
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing
Start with the Jupyter notebook for data preprocessing:
```bash
jupyter notebook data/preprocessing.ipynb
```

### 2. Clustering Analysis
Run the main clustering optimization script:
```bash
cd data
python clustering.py
```

This will:
- Load the preprocessed data
- Run hyperparameter optimization for all clustering algorithms
- Generate results in `clustering_results_1.csv`

### 3. Feature Selection
Perform automated feature selection:
```bash
cd data
python feature_selection.py
```

This will:
- Load clustering results
- Apply greedy feature elimination
- Update results with optimal feature combinations

### 4. Results Visualization
Analyze and visualize results:
```bash
jupyter notebook data/visualization_results.ipynb
```

## Methodology

### Hyperparameter Optimization
The project uses **Optuna** for Bayesian optimization of clustering algorithm hyperparameters. Each algorithm is optimized across multiple dimensions including:
- Algorithm-specific parameters
- Dimensionality reduction settings
- Preprocessing options

### Feature Selection
Implements a **greedy backward elimination** algorithm that:
1. Starts with all features
2. Iteratively removes features that least impact clustering quality
3. Stops when removal significantly degrades performance
4. Maintains a configurable quality threshold

### Evaluation Strategy
- **Multi-metric evaluation** using appropriate metrics for each algorithm
- **Penalty system** for imbalanced or extreme cluster distributions
- **Cross-validation** approach for robust results

## Key Parameters

### Clustering Optimization
- `n_trials`: Number of optimization trials (default: 10,000)
- Algorithm-specific hyperparameter ranges defined in `objective()` function

### Feature Selection
- `threshold`: Quality degradation threshold (default: 0.05)
- `scaling`: Whether to apply feature scaling (default: True)
- `n_jobs`: Number of parallel jobs (default: -1)

## Output Files

- `clustering_results_1.csv`: Complete clustering results with hyperparameters
- `cluster_means.xlsx`: Cluster centroid analysis
- `*.png`: Visualization outputs (heatmaps, convex hulls, cluster plots)

## Requirements

See `requirements.txt` for complete dependency list. Key packages:
- scikit-learn (machine learning algorithms)
- optuna (hyperparameter optimization)
- umap-learn (dimensionality reduction)
- pandas/numpy (data manipulation)
- hdbscan (density-based clustering)

## Contributing

This project continues work from the YueyOare/coursework repository. When contributing:
1. Follow the established code structure
2. Document new algorithms or methods
3. Update requirements.txt for new dependencies
4. Add appropriate tests for new functionality

## License

[Specify your license here]

## Citation

If you use this work in your research, please cite:
```
[Add appropriate citation format]
```

## Contact

[Add contact information]