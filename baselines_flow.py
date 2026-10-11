"""Flow-level one-class baselines aggregated to the same 5-minute windows (same split, same threshold rule).

1. Sample <= 300 BENIGN flows per FIT window (FIT+CAL roles) and per VAL window.
2. Preprocess (signed log1p, median imputation, RobustScaler 25-75, fitted on the FIT sample) and fit
   IsolationForest(100) on 200k, KMeans(20) on 100k, LOF(novelty, k=20) on 10k, OCSVM(rbf, nu=0.1) on 5k flows.
   Flow threshold for the 'share' aggregate = Q99 of the VAL benign-sample scores.
3. Score every flow of the VAL and TEST days; window aggregates per model: max, q99, top10 mean, mean, share above
   the flow threshold.
Labels are used only to pick BENIGN flows for fitting (one-class training on benign traffic).
Usage: python baselines_flow.py <dataset> <csv_dir> <data_dir> <split.csv> <out_dir> [--workers 32]
Writes <out_dir>/window_scores.parquet (graph_id + FLOW_<model>_<agg>), models.joblib, DONE.
"""
import argparse, json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.csv as pc
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import DATASETS

NON_FEATURE = {'id', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label', 'Attempted Category'}
SAMPLE_PER_WINDOW, SEED = 300, 42
NAMES = ['IF', 'KMEANS', 'LOF', 'OCSVM']


def read_day(path, window='5min'):
    t = pc.read_csv(path, read_options=pc.ReadOptions(block_size=256 << 20))
    cols = [c for c in t.schema.names if c not in NON_FEATURE]
    X = np.column_stack([np.asarray(t.column(c).to_numpy(zero_copy_only=False), dtype=np.float32) for c in cols])
    ws = pd.to_datetime(t.column('Timestamp').to_pandas(), format='ISO8601').dt.floor(window).to_numpy()
    lab = t.column('Label').to_pandas().astype(str).str.strip().to_numpy()
    return cols, X, ws, lab


def prep(X, med, sc):
    X = X.astype(np.float64); X[~np.isfinite(X)] = np.nan
    X = np.sign(X) * np.log1p(np.abs(X)); X = np.where(np.isnan(X), med, X)
    return sc.transform(X)


def score_chunk(z, models):
    return np.column_stack([-models['IF'].decision_function(z), models['KMEANS'].transform(z).min(axis=1),
                            -models['LOF'].decision_function(z), -models['OCSVM'].decision_function(z)]).astype(np.float32)


def score_rows(Xc, med, sc, models):
    return score_chunk(prep(Xc, med, sc), models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset'); ap.add_argument('csv_dir'); ap.add_argument('data'); ap.add_argument('split'); ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(dict(vars(a), split=str(a.split)), indent=1))
    cfg = DATASETS[a.dataset]
    meta = pd.read_parquet(Path(a.data) / 'meta.parquet'); meta['win_start'] = pd.to_datetime(meta['win_start'])
    win = f"{int(round((pd.to_datetime(meta.win_end) - meta.win_start).median().total_seconds() / 60))}min" if 'win_end' in meta else '5min'
    role = pd.read_csv(a.split).set_index('graph_id').role
    meta['role'] = meta.graph_id.map(role)
    rng = np.random.default_rng(SEED)

    # 1. benign samples from FIT and VAL windows
    samples = {'FIT': [], 'VAL': []}
    cols = None
    for day in cfg['days']:
        m = meta[meta.source_file.eq(day) & meta.role.isin(['FIT', 'CAL', 'VAL'])]
        if not len(m):
            continue
        t = time.time(); cols, X, ws, lab = read_day(Path(a.csv_dir) / day, win)
        for r, rs in (('FIT', ['FIT', 'CAL']), ('VAL', ['VAL'])):
            keep = set(m.loc[m.role.isin(rs), 'win_start'].to_numpy())
            idx = np.flatnonzero((lab == 'BENIGN') & pd.Series(ws).isin(keep).to_numpy())
            if not len(idx):
                continue
            key = rng.random(len(idx))
            s = pd.DataFrame(dict(ws=ws[idx], key=key, row=idx)).sort_values('key').groupby('ws').head(SAMPLE_PER_WINDOW)
            samples[r].append(X[s.row.to_numpy()])
        print(f'sample {day}: {time.time() - t:.0f}s', flush=True)
        del X
    Xf, Xv = np.vstack(samples['FIT']).astype(np.float64), np.vstack(samples['VAL']).astype(np.float64)
    Xf[~np.isfinite(Xf)] = np.nan
    Lf = np.sign(Xf) * np.log1p(np.abs(Xf))
    med = np.nanmedian(Lf, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    sc = RobustScaler(quantile_range=(25, 75)).fit(np.where(np.isnan(Lf), med, Lf))
    Zf, Zv = prep(Xf, med, sc), prep(Xv, med, sc)
    sub = lambda n: Zf[rng.choice(len(Zf), min(n, len(Zf)), replace=False)]
    models = dict(IF=IsolationForest(n_estimators=100, random_state=SEED, n_jobs=a.workers).fit(sub(200_000)),
                  KMEANS=KMeans(n_clusters=20, n_init=3, random_state=SEED).fit(sub(100_000)),
                  LOF=LocalOutlierFactor(n_neighbors=20, novelty=True).fit(sub(10_000)),
                  OCSVM=OneClassSVM(kernel='rbf', gamma='scale', nu=0.1).fit(sub(5_000)))
    for m in models.values():
        if hasattr(m, 'n_jobs'):
            m.n_jobs = 1
    sv = score_chunk(Zv, models)
    thr = {n: float(np.quantile(sv[:, i], 0.99)) for i, n in enumerate(NAMES)}
    joblib.dump(dict(cols=cols, median=med, scaler=sc, models=models, flow_thr_q99_val=thr), out / 'models.joblib', compress=3)
    print('models fitted', dict(fit_flows=len(Zf), val_flows=len(Zv), thr=thr), flush=True)

    # 3. score VAL and TEST days
    rows = []
    for day in cfg['days']:
        m = meta[meta.source_file.eq(day) & meta.role.isin(['VAL', 'TEST', 'TEST_NEAR', 'TEST_ATTEMPTED'])]
        if not len(m):
            continue
        t = time.time(); _, X, ws, _ = read_day(Path(a.csv_dir) / day, win)
        bounds = list(range(0, len(X), 200_000)) + [len(X)]
        S = np.vstack(Parallel(n_jobs=a.workers)(delayed(score_rows)(X[s:e], med, sc, models)
                                                for s, e in zip(bounds[:-1], bounds[1:])))
        df = pd.DataFrame(S, columns=NAMES); df['win_start'] = ws
        wanted = set(m.win_start.to_numpy())
        for w, g in df[df.win_start.isin(wanted)].groupby('win_start'):
            r = dict(source_file=day, win_start=w, n_flows_scored=len(g))
            for n in NAMES:
                v = g[n].to_numpy()
                r[f'FLOW_{n}_max'] = float(v.max()); r[f'FLOW_{n}_q99'] = float(np.quantile(v, 0.99))
                r[f'FLOW_{n}_mean'] = float(v.mean()); r[f'FLOW_{n}_top10'] = float(np.sort(v)[-10:].mean())
                r[f'FLOW_{n}_share'] = float((v > thr[n]).mean())
            rows.append(r)
        print(f'scored {day}: {len(X):,} flows, {time.time() - t:.0f}s', flush=True)
        del X, S, df
    R = pd.DataFrame(rows)
    R['win_start'] = pd.to_datetime(R['win_start'])
    R = meta[['graph_id', 'source_file', 'win_start']].merge(R, on=['source_file', 'win_start'], how='inner')
    R.drop(columns=['source_file', 'win_start']).to_parquet(out / 'window_scores.parquet', index=False)
    (out / 'DONE').write_text(json.dumps(dict(windows=len(R), thr=thr)))
    print('DONE', len(R), 'windows', flush=True)


if __name__ == '__main__':
    main()
