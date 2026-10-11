"""Classic one-class baselines on window-level vectors of the SAME graphs (same split, same threshold rule).

Feature sets: G188 = full graph_attr (47 current-window features + delta to previous window + z-score and ratio
vs the previous 12 windows); G47 = the 47 current-window features only.
Preprocessing: signed log1p, RobustScaler fitted on FIT+CAL windows, clip +-50.
Models: IsolationForest, LOF (novelty), OCSVM (RBF, nu=0.1), GMM (4, diag), PCA reconstruction (95% var),
KMeans(8) distance, MLP autoencoder (32-8-32). Higher score = more anomalous. Labels are not read.
Usage: python baselines_graph.py <data_dir> <split.csv> <out_dir> [--seed 42]
Writes <out_dir>/window_scores.parquet (graph_id + one column per feature set x model), DONE.
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings('ignore')


def models(seed):
    return {
        'IF': lambda: IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1),
        'LOF': lambda: LocalOutlierFactor(n_neighbors=20, novelty=True),
        'OCSVM': lambda: OneClassSVM(nu=0.1, gamma='scale'),
        'GMM': lambda: GaussianMixture(n_components=4, covariance_type='diag', reg_covar=1e-3, random_state=seed),
        'PCA': lambda: PCA(n_components=0.95, random_state=seed),
        'KMeans': lambda: KMeans(n_clusters=8, n_init=10, random_state=seed),
        'MLP-AE': lambda: MLPRegressor(hidden_layer_sizes=(32, 8, 32), max_iter=2000, early_stopping=True,
                                       validation_fraction=0.15, random_state=seed),
    }


def score(name, m, z):
    if name == 'GMM':
        return -m.score_samples(z)
    if name == 'PCA':
        return ((z - m.inverse_transform(m.transform(z))) ** 2).mean(1)
    if name == 'KMeans':
        return m.transform(z).min(1)
    if name == 'MLP-AE':
        return ((z - m.predict(z)) ** 2).mean(1)
    return -m.decision_function(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('split'); ap.add_argument('out'); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(dict(vars(a), split=str(a.split)), indent=1))
    graphs = torch.load(Path(a.data) / 'graphs.pt', map_location='cpu', weights_only=False, mmap=True)
    X = np.vstack([g['graph_attr'].float().view(-1).numpy() for g in graphs]).astype(np.float64)
    del graphs
    role = pd.read_csv(a.split).set_index('graph_id').role.reindex(range(len(X))).to_numpy()
    fit = np.isin(role, ['FIT', 'CAL'])
    res = pd.DataFrame(dict(graph_id=np.arange(len(X))))
    for fs, cols in (('G188', slice(None)), ('G47', slice(0, 47))):
        Z = X[:, cols]
        Z = np.sign(Z) * np.log1p(np.abs(Z)); Z[~np.isfinite(Z)] = 0.0
        sc = RobustScaler().fit(Z[fit]); Z = np.clip(sc.transform(Z), -50, 50)
        for name, ctor in models(a.seed).items():
            t = time.time(); m = ctor()
            m.fit(Z[fit], Z[fit]) if name == 'MLP-AE' else m.fit(Z[fit])
            res[f'{fs}_{name}'] = score(name, m, Z)
            print(f'{fs} {name}: {time.time() - t:.1f}s', flush=True)
    res.to_parquet(out / 'window_scores.parquet', index=False)
    (out / 'DONE').write_text(json.dumps(dict(fit_windows=int(fit.sum()), features=X.shape[1])))


if __name__ == '__main__':
    main()
