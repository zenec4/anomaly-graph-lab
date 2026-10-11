"""Summarise the leave-one-day-out runs (reads labels; nothing here feeds back into the models).

Usage: python lodo_summary.py <data_dir> <runs_lodo_dir> <out_dir>
For every fold (= test day) and score: threshold = Q85 of the fold's VAL windows; detections are pooled over folds.
Pooled metrics per (method, score):
  macro_f1 / recall / fpr          pooled TEST windows of all folds, each judged with its own fold threshold
  auc_within_day                   P(attack window scores above a benign window of the SAME test day), i.e. the fold
                                   AUCs weighted by n_attack x n_benign (insensitive to day-level score shifts)
  auc_pooled                       AUC over all folds after robust per-fold normalisation by the VAL windows
                                   ((s - median_VAL) / IQR_VAL); includes day-to-day shifts
  tpr_fpr1 / tpr_fpr5              on the pooled normalised scores
  macro_f1_ci / auc_pooled_ci      block bootstrap over (day, hour) blocks, 1000 draws (main scores only)
Outputs: per_fold.csv, pooled.csv, pooled_by_seed.csv (mean/std over seeds), recall_by_type.csv
"""
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

MAIN = {'S_grpd', 'n_flows', 'FLOW_LOF_share', 'FLOW_LOF_mean', 'FLOW_OCSVM_top10', 'G188_IF', 'G47_PCA'}


def tpr_at(y, s, f):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= f + 1e-12].max())


def main():
    data, runs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    L = pd.read_parquet(data / 'labels.parquet'); M = pd.read_parquet(data / 'meta.parquet')
    D = L[['graph_id', 'source_file', 'win_start', 'has_attack_strict', 'attack_type']].merge(M[['graph_id', 'n_flows_raw']], on='graph_id')
    D['hour'] = pd.to_datetime(D.win_start).dt.hour; D['y'] = D.has_attack_strict.astype(int)
    entries = []          # (fold, method, seed, score-column, scores array aligned with D)
    for r in sorted(p for p in runs.iterdir() if (p / 'DONE').exists()):
        fold, method = r.name.split('__', 1)
        W = D[['graph_id']].merge(pd.read_parquet(r / 'window_scores.parquet'), on='graph_id', how='left')
        m = re.match(r'(.+)_s(\d+)$', method)
        base, seed = (m.group(1), int(m.group(2))) if m else (method, 0)
        cols = ['S_grpd', 'S_residual_entity_gate', 'S_local_subgraph', 'gae_loss'] if 'S_grpd' in W else \
               [c for c in W.columns if c not in ('graph_id', 'n_flows_scored')]
        for c in cols:
            entries.append((fold, base, seed, c, W[c].to_numpy(dtype=float)))
    for fold in sorted({e[0] for e in entries}):
        entries.append((fold, 'confound', 0, 'n_flows', D.n_flows_raw.to_numpy(dtype=float)))
    roles = {f: pd.read_csv(data / f'split_lodo_{f}.csv').set_index('graph_id').role.reindex(D.graph_id).to_numpy()
             for f in {e[0] for e in entries}}

    per_fold, pooled_parts = [], {}
    for fold, method, seed, col, s in entries:
        role = roles[fold]; val, test = role == 'VAL', role == 'TEST'
        if np.isnan(s[val | test]).any():
            continue
        thr = float(np.quantile(s[val], 0.85))
        med, iqr = np.median(s[val]), np.subtract(*np.percentile(s[val], [75, 25]))
        y = D.y.to_numpy()[test]; st = s[test]; p = (st > thr).astype(int)
        both = 0 < y.sum() < len(y)
        per_fold.append(dict(fold=fold, method=method, seed=seed, score=col, n_attack=int(y.sum()), n_benign=int((y == 0).sum()),
                             auc=roc_auc_score(y, st) if both else np.nan, recall=float(p[y == 1].mean()) if y.sum() else np.nan,
                             fpr=float(p[y == 0].mean()) if (y == 0).any() else np.nan, threshold=thr))
        part = pd.DataFrame(dict(fold=fold, y=y, s=st, z=(st - med) / max(iqr, 1e-12), det=p, hour=D.hour.to_numpy()[test],
                                 attack_type=D.attack_type.to_numpy()[test]))
        pooled_parts.setdefault((method, seed, col), []).append(part)
    PF = pd.DataFrame(per_fold); PF.to_csv(out / 'per_fold.csv', index=False)

    rng = np.random.default_rng(7); rows, types = [], []
    for (method, seed, col), parts in pooled_parts.items():
        P = pd.concat(parts, ignore_index=True); y = P.y.to_numpy()
        f = PF[(PF.method == method) & (PF.seed == seed) & (PF.score == col) & PF.auc.notna()]
        w = f.n_attack * f.n_benign
        row = dict(method=method, seed=seed, score=col, folds=len(parts), n_attack=int(y.sum()), n_benign=int((y == 0).sum()),
                   macro_f1=f1_score(y, P.det, average='macro'), recall=float(P.det[y == 1].mean()), fpr=float(P.det[y == 0].mean()),
                   auc_within_day=float((f.auc * w).sum() / w.sum()) if len(f) else np.nan, auc_pooled=roc_auc_score(y, P.z),
                   tpr_fpr1=tpr_at(y, P.z, .01), tpr_fpr5=tpr_at(y, P.z, .05))
        if col in MAIN:
            blocks = (P.fold + '_' + P.hour.astype(str)).to_numpy(); ub = np.unique(blocks)
            idx = {b: np.flatnonzero(blocks == b) for b in ub}; F1, AU = [], []
            for _ in range(1000):
                i = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
                if 0 < y[i].sum() < len(i):
                    F1.append(f1_score(y[i], P.det.to_numpy()[i], average='macro')); AU.append(roc_auc_score(y[i], P.z.to_numpy()[i]))
            row['macro_f1_ci'] = np.percentile(F1, [2.5, 97.5]).round(3).tolist(); row['auc_pooled_ci'] = np.percentile(AU, [2.5, 97.5]).round(3).tolist()
        rows.append(row)
        for t, g in P[P.y == 1].groupby('attack_type'):
            types.append(dict(method=method, seed=seed, score=col, attack_type=t, n=len(g), recall=g.det.mean()))
    R = pd.DataFrame(rows).sort_values(['score', 'method', 'seed']); R.to_csv(out / 'pooled.csv', index=False)
    pd.DataFrame(types).to_csv(out / 'recall_by_type.csv', index=False)
    num = ['macro_f1', 'recall', 'fpr', 'auc_within_day', 'auc_pooled', 'tpr_fpr1', 'tpr_fpr5']
    G = R.groupby(['method', 'score'])[num].agg(['mean', 'std', 'count']); G.columns = ['_'.join(c) for c in G.columns]
    G = G.reset_index().sort_values('auc_pooled_mean', ascending=False); G.to_csv(out / 'pooled_by_seed.csv', index=False)
    pd.set_option('display.width', 250)
    print(G[['method', 'score', 'macro_f1_mean', 'macro_f1_std', 'auc_within_day_mean', 'auc_pooled_mean', 'auc_pooled_std',
             'tpr_fpr5_mean', 'fpr_mean']].round(3).head(25).to_string(index=False))


if __name__ == '__main__':
    main()
