import numpy as np
import optuna
import pandas as pd
import umap.umap_ as umap
from joblib import Parallel, delayed
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
                             Birch, MeanShift, AffinityPropagation, OPTICS, HDBSCAN)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")


def cluster_penalty(labels, n_samples):
    valid = labels[labels >= 0]
    if valid.size == 0:
        return 100
    counts = np.bincount(valid)
    return sum(1 for c in counts if c < 0.05 * n_samples or c > 0.8 * n_samples)


def cluster_evaluation(X, model, method_name, reduction_method=None, reduction_params=None):
    try:
        if reduction_method == 'PCA':
            pca = PCA(n_components=reduction_params, random_state=42)
            X = pca.fit_transform(X)
        elif reduction_method == 'UMAP':
            umap_model = umap.UMAP(n_components=reduction_params, random_state=42)
            X = umap_model.fit_transform(X)

        n_samples = X.shape[0]
        if method_name == 'GaussianMixture':
            labels = model.fit(X).predict(X)
            metric = 'likelihood'
            raw = model.score(X)
        else:
            labels = model.fit_predict(X)
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2 or len(unique_labels) >= n_samples:
                return -1e6, 'error', reduction_method, reduction_params

            if method_name in ['KMeans', 'Birch', 'HDBSCAN']:
                metric = 'calinski_harabasz_score'
                raw = calinski_harabasz_score(X, labels)
            elif method_name in ['DBSCAN', 'AgglomerativeClustering', 'SpectralClustering',
                                 'MeanShift', 'AffinityPropagation', 'OPTICS']:
                metric = 'silhouette_score'
                raw = silhouette_score(X, labels)
            else:
                metric = 'silhouette_score'
                raw = silhouette_score(X, labels)

        return raw - cluster_penalty(labels, n_samples), metric, reduction_method, reduction_params

    except Exception:
        return -1e6, 'error', reduction_method, reduction_params


def apply_dimensionality_reduction(X, method, trial):
    if method == 'PCA':
        n_components = trial.suggest_int('pca_n_components', 2, X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(X)
    elif method == 'UMAP':
        n_components = trial.suggest_int('umap_n_components', 2, X.shape[1])
        n_neighbors = trial.suggest_int('umap_n_neighbors', 5, 50)
        min_dist = trial.suggest_float('umap_min_dist', 0.001, 0.9)
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        return umap_model.fit_transform(X)
    return X


def objective(trial, X, method):
    # Выбор метода для уменьшения размерности
    dim_reduction_method = trial.suggest_categorical('dimensionality_reduction', ['None', 'PCA', 'UMAP'])

    X_reduced = apply_dimensionality_reduction(X, dim_reduction_method, trial)

    if method == 'KMeans':
        params = {
            'n_clusters': trial.suggest_int('n_clusters', 3, 50),
            'init': trial.suggest_categorical('init', ['k-means++', 'random']),
            'max_iter': trial.suggest_int('max_iter', 100, 2000),
            'n_init': trial.suggest_int('n_init', 5, 100),
            'algorithm': trial.suggest_categorical('algorithm', ['lloyd', 'elkan']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True)
        }
        model = KMeans(**params, random_state=42)

    elif method == 'DBSCAN':
        params = {
            'eps': trial.suggest_float('eps', 0.001, 5.0, log=True),
            'min_samples': trial.suggest_int('min_samples', 2, 200),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine', 'chebyshev']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': trial.suggest_int('leaf_size', 10, 100)
        }
        model = DBSCAN(**params)

    elif method == 'GaussianMixture':
        params = {
            'n_components': trial.suggest_int('n_components', 2, min(X.shape[0] // 5, 20)),
            'covariance_type': trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical']),
            'reg_covar': trial.suggest_float('reg_covar', 1e-5, 1e-1, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'n_init': trial.suggest_int('n_init', 1, 20)
        }
        try:
            model = GaussianMixture(**params, random_state=42)
        except Exception:
            return -1e6

    elif method == 'AgglomerativeClustering':
        params = {
            'n_clusters': trial.suggest_int('n_clusters', 3, 50),
            'linkage': trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single']),
            'compute_distances': trial.suggest_categorical('compute_distances', [True, False])
        }
        model = AgglomerativeClustering(**params)

    elif method == 'SpectralClustering':
        params = {
            'n_clusters': trial.suggest_int('n_clusters', 3, 50),
            'affinity': trial.suggest_categorical('affinity', ['nearest_neighbors', 'rbf']),
            'assign_labels': trial.suggest_categorical('assign_labels', ['kmeans', 'discretize']),
            'random_state': trial.suggest_int('random_state', 0, 100)
        }
        model = SpectralClustering(**params)

    elif method == 'Birch':
        params = {
            'threshold': trial.suggest_float('threshold', 0.01, 5.0, log=True),
            'branching_factor': trial.suggest_int('branching_factor', 10, 200),
            'n_clusters': trial.suggest_int('n_clusters', 3, 50),
            'compute_labels': trial.suggest_categorical('compute_labels', [True, False])
        }
        model = Birch(**params)

    elif method == 'MeanShift':
        params = {
            'bandwidth': trial.suggest_float('bandwidth', 0.01, 20.0, log=True),
            'bin_seeding': trial.suggest_categorical('bin_seeding', [True, False]),
            'max_iter': trial.suggest_int('max_iter', 100, 1000)
        }
        model = MeanShift(**params)

    elif method == 'AffinityPropagation':
        params = {
            'damping': trial.suggest_float('damping', 0.5, 0.99),
            'max_iter': trial.suggest_int('max_iter', 200, 2000),
            'preference': trial.suggest_float('preference', -1000, 0),
            'convergence_iter': trial.suggest_int('convergence_iter', 10, 100)
        }
        model = AffinityPropagation(**params)

    elif method == 'OPTICS':
        params = {
            'min_samples': trial.suggest_int('min_samples', 2, 200),
            'xi': trial.suggest_float('xi', 0.01, 0.3),
            'min_cluster_size': trial.suggest_int('min_cluster_size', 2, 200),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine', 'chebyshev']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        }
        model = OPTICS(**params)

    elif method == 'HDBSCAN':
        params = {
            'min_cluster_size': trial.suggest_int('min_cluster_size', 5, 200),
            'min_samples': trial.suggest_int('min_samples', 1, 100),
            'cluster_selection_epsilon': trial.suggest_float('cluster_selection_epsilon', 0.0, 5.0),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine', 'chebyshev'])
        }
        model = HDBSCAN(**params)

    else:
        raise ValueError(f'Unknown method: {method}')

    return cluster_evaluation(X_reduced, model, method)


def run_experiment(X, method, n_trials=1000):
    warnings.filterwarnings("ignore")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, method)[0], n_trials=n_trials)
    best_value, metric, reduction_method, reduction_params = objective(study.best_trial, X, method)
    best_params = study.best_trial.params
    return best_params, best_value, metric, reduction_method, reduction_params


if __name__ == '__main__':
    df = pd.read_csv('preprocessed/prepared_data.csv')
    X = df.drop(columns=['КОД3 основной']).values

    methods = ['KMeans', 'DBSCAN', 'GaussianMixture', 'AgglomerativeClustering',
               'SpectralClustering', 'Birch', 'MeanShift', 'AffinityPropagation',
               'OPTICS', 'HDBSCAN']
    exp_results = []

    results = Parallel(n_jobs=-1)(delayed(run_experiment)(X, m) for m in methods)
    for method, (params, score, metric, reduction_method, reduction_params) in zip(methods, results):
        exp_results.append(
            {'method': method, 'params': params, 'score': score, 'metric': metric, 'reduction_method': reduction_method,
             'reduction_params': reduction_params})

    pd.DataFrame(exp_results).to_csv('clustering_results.csv', index=False)
