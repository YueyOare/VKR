import warnings
from ast import literal_eval

import numpy as np
import pandas as pd
import umap
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, SpectralClustering, Birch, MeanShift, \
    AffinityPropagation, OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
results_filename = 'clustering_results_1.csv'
results = pd.read_csv(results_filename)
df = pd.read_csv('preprocessed/prepared_data_nonumap.csv')

methods_map = {
    'KMeans': KMeans, 'DBSCAN': DBSCAN, 'HDBSCAN': HDBSCAN, 'AgglomerativeClustering': AgglomerativeClustering,
    'SpectralClustering': SpectralClustering, 'Birch': Birch, 'MeanShift': MeanShift,
    'AffinityPropagation': AffinityPropagation, 'GaussianMixture': GaussianMixture,
    'OPTICS': OPTICS
}


def reduce_and_scale(X, params, scale=True):
    method = params.pop('dimensionality_reduction', None)
    if method == 'PCA':
        reducer = PCA(n_components=params.pop('pca_n_components', 2), random_state=42)
        X = reducer.fit_transform(X)
    elif method == 'UMAP':
        reducer = umap.UMAP(
            n_components=params.pop('umap_n_components', 2),
            n_neighbors=params.pop('umap_n_neighbors', 15),
            min_dist=params.pop('umap_min_dist', 0.1),
            random_state=42
        )
        X = reducer.fit_transform(X)
    X = StandardScaler().fit_transform(X) if scale else X
    return X


def cluster_and_label(method, params, X):
    model_cls = methods_map[method]
    if model_cls is None:
        raise ValueError(f'Method {method} is not supported in this setup.')

    params.pop('dimensionality_reduction', None)
    params = {k: v for k, v in params.items() if not k.startswith(('umap', 'pca'))}

    if method == 'GaussianMixture':
        model = model_cls(**params, random_state=42)
        labels = model.fit(X).predict(X)
    else:
        model = model_cls(**params)
        labels = model.fit_predict(X)
    return labels, model


def calculate_metric(X, labels, metric_name, model=None):
    if metric_name == 'silhouette_score':
        return silhouette_score(X, labels)
    elif metric_name == 'calinski_harabasz_score':
        return calinski_harabasz_score(X, labels)
    elif metric_name == 'likelihood':
        return np.mean(model.score(X))
    else:
        raise ValueError(f"Метрика {metric_name} не поддерживается.")


def process_row(idx, row, X_full, feature_names, scaling=True, threshold=0.05):
    warnings.filterwarnings("ignore")
    method = row['method']
    params = literal_eval(row['params'])
    metric = row['metric']
    features = list(range(X_full.shape[1]))
    best_comb = []

    try:
        X_base = reduce_and_scale(X_full[:, features], params.copy(), scale=scaling)
        labels, model = cluster_and_label(method, params.copy(), X_base)
        if len(np.unique(labels)) < 2:
            print(f'{idx} {method}: один кластер')
            return idx, None, None
        best_score = calculate_metric(X_base, labels, metric, model)
        print(f'{idx} {method}: базовый {metric} = {best_score:.4f}')
    except Exception as e:
        print(f'{idx} {method} пропущен: {e}')
        return idx, None, None

    step = 1
    while True:
        best_f = None
        current_best_score = (1 - threshold) * best_score

        for f in features:
            remaining = [i for i in features if i != f]
            X_try = X_full[:, remaining]
            try:
                X_reduced = reduce_and_scale(X_try, params.copy(), scale=scaling)
                labels, model = cluster_and_label(method, params.copy(), X_reduced)
                if len(np.unique(labels)) < 2:
                    continue
                score = calculate_metric(X_reduced, labels, metric, model)
            except:
                continue

            if score > current_best_score:
                current_best_score = score
                best_f = f

        if best_f is None:
            print(
                f'{idx} {method}: остановка, итерация {step}, score = {best_score:.4f}, удалено признаков: {len(best_comb)}')
            break

        print(
            f'{idx} {method}: итерация {step}, удалён признак {feature_names[best_f]}, score = {current_best_score:.4f}')
        features.remove(best_f)
        best_comb.append(best_f)
        best_score = current_best_score
        step += 1

    return idx, [feature_names[i] for i in best_comb], best_score


def greedy_feature_elimination_parallel(results, df, scaling=True, threshold=0.05, n_jobs=-1):
    X_full = df.drop(columns=['КОД3 основной']).values
    feature_names = df.drop(columns=['КОД3 основной']).columns.tolist()
    results['best_comb'] = None
    results['best_score'] = None

    processed = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(idx, row, X_full, feature_names, scaling, threshold)
        for idx, row in results.iterrows()
    )

    for idx, best_comb, best_score in processed:
        if best_comb is not None:
            results.at[idx, 'best_comb'] = str(best_comb)
            results.at[idx, 'best_score'] = best_score

    return results


results = greedy_feature_elimination_parallel(results, df)
results.to_csv(results_filename, index=False)
