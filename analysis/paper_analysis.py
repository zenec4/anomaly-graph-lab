"""Secondary analyses of the paper, computed from the outputs of run_all.py.

Usage: python analysis/paper_analysis.py --root <run_all root> [--out analysis_out]

Writes CSV/JSON files to --out:
  config_selection.csv   baselines: configuration with the best test AUC vs. one fixed rule (isolation forest wherever
                         a detector must be chosen, top-10 mean wherever entity scores are aggregated) and the median
                         over configurations (day-disjoint and LODO, 5-min windows)
  per_attack_<ds>.csv    recall per attack type at a matched 10% FPR under LODO (threshold on the pooled benign test
                         windows; mean over seeds), and the number of types with recall >= 0.5 per detector
  fold_groups.json       CSE-CIC-IDS2018 LODO pooled AUC on the 3 folds of the day-disjoint test days vs. the other 7
  paired_bootstrap.json  paired block bootstrap (day x hour blocks, 1000 resamples) of AUC differences between detectors
  attempted_2017.json    CIC-IDS2017 day-disjoint AUC with Attempted-only windows excluded / counted as attack / as benign
"""
import argparse, glob, json, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FAMILIES = ['anomale', 'pcarec', 'dominant', 'ocgnn', 'bl_flow', 'bl_graph']
FIXED = dict(anomale='ANOMALE_IF_top10', pcarec='PCAREC_REC_top10', dominant='DOMINANT_top10', ocgnn='OCGNN_top10',
             bl_flow='FLOW_IF_top10', bl_graph='G188_IF')
DETECTORS = ['grpd', 'pcarec', 'anomale', 'dominant', 'ocgnn', 'nflows']
SEEDS = dict(grpd=(42, 43, 44), anomale=(42, 43, 44), dominant=(42, 43, 44), ocgnn=(42, 43, 44), pcarec=(None,), nflows=(None,))
DAY_SEEDS = dict(SEEDS, grpd=(42, 43, 44, 45, 46))
TEST_DAYS_2018 = {'Wednesday-28-02-2018', 'Thursday-01-03-2018', 'Friday-02-03-2018'}


def best_configs(root, ds):
    """Configuration of each multi-configuration detector with the best test AUC (as reported in the tables)."""
    M = pd.read_csv(root / ds / 'eval' / 'day' / 'metrics_by_seed.csv')
    P = pd.read_csv(root / ds / 'eval' / 'lodo' / 'pooled_by_seed.csv')
    day = {f: M[M.group == f].sort_values('roc_auc_mean').score.iloc[-1] for f in FAMILIES if (M.group == f).any()}
    lodo = {f: P[P.method == f].sort_values('auc_pooled_mean').score.iloc[-1] for f in FAMILIES if (P.method == f).any()}
    return day, lodo, M, P


def config_selection(root, ds, M, P):
    rows = []
    for key, T, col, grp in [('day', M, 'roc_auc_mean', 'group'), ('lodo', P, 'auc_pooled_mean', 'method')]:
        for f in FAMILIES:
            G = T[T[grp] == f]
            if not len(G):
                continue
            fx = G[G.score == FIXED[f]][col]
            rows.append(dict(dataset=ds, test=key, detector=f, n_configs=len(G), best=G[col].max(),
                             fixed=float(fx.iloc[0]) if len(fx) else np.nan, median=G[col].median()))
    return rows


def lodo_scores(root, ds, det, seed, cols):
    data = root / ds / 'graphs' / 'data'
    D = pd.read_parquet(data / 'labels.parquet'); D['hour'] = pd.to_datetime(D.win_start).dt.hour
    parts = []
    for fc in sorted(glob.glob(str(data / 'split_lodo_*.csv'))):
        fold = re.search(r'split_lodo_(.+)\.csv', fc).group(1)
        role = pd.read_csv(fc).set_index('graph_id').role.reindex(D.graph_id).to_numpy()
        if det == 'nflows':
            x = D.n_flows_raw.to_numpy(float)
        else:
            name = f'{fold}__{det}' + ('' if seed is None else f'_s{seed}')
            W = pd.read_parquet(root / ds / 'runs' / 'lodo' / name / 'window_scores.parquet')
            x = D[['graph_id']].merge(W, on='graph_id', how='left')[cols[det]].to_numpy(float)
        val, test = role == 'VAL', role == 'TEST'
        med, iqr = np.median(x[val]), np.subtract(*np.percentile(x[val], [75, 25]))
        parts.append(pd.DataFrame(dict(fold=fold, z=(x[test] - med) / max(iqr, 1e-12), y=D.has_attack_strict.to_numpy()[test],
                                       t=D.attack_type.to_numpy()[test], blk=[f'{fold}_{h}' for h in D.hour.to_numpy()[test]])))
    return pd.concat(parts, ignore_index=True)


def day_scores(root, ds, det, seed, cols):
    data = root / ds / 'graphs' / 'data'
    L = pd.read_parquet(data / 'labels.parquet')
    role = pd.read_csv(data / 'split_day.csv').set_index('graph_id').role.reindex(L.graph_id).to_numpy()
    if det == 'nflows':
        x = L.n_flows_raw.to_numpy(float)
    else:
        name = det + ('' if seed is None else f'_s{seed}')
        W = pd.read_parquet(root / ds / 'runs' / 'day' / name / 'window_scores.parquet')
        x = L[['graph_id']].merge(W, on='graph_id', how='left')[cols[det]].to_numpy(float)
    blk = (L.source_file.astype(str) + '_' + pd.to_datetime(L.win_start).dt.hour.astype(str)).to_numpy()
    return pd.DataFrame(dict(role=role, z=x, y=L.has_attack_strict.to_numpy(), blk=blk))


def recall_at(P, fpr=0.10):
    thr = np.quantile(P.z[P.y == 0], 1 - fpr); A = P[P.y == 1]
    return A.assign(det=A.z > thr).groupby('t').det.agg(['size', 'mean'])


def boot_diff(A, B, n=1000, seed=0):
    """Mean over seeds of AUC(A) minus mean over seeds of AUC(B), with a paired block-bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    blk = A[0].blk.to_numpy(); ub = np.unique(blk); idx = {b: np.flatnonzero(blk == b) for b in ub}
    y = A[0].y.to_numpy(); za = [p.z.to_numpy() for p in A]; zb = [p.z.to_numpy() for p in B]
    auc = lambda ii, zs: np.mean([roc_auc_score(y[ii], z[ii]) for z in zs])
    allidx = np.arange(len(y)); obs = auc(allidx, za) - auc(allidx, zb); ds = []
    for _ in range(n):
        ii = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        if 0 < y[ii].sum() < len(ii):
            ds.append(auc(ii, za) - auc(ii, zb))
    return [float(obs), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', default='analysis_out')
    ap.add_argument('--datasets', nargs='+', default=['cic2018', 'cic2017'])
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg_rows, boot, LS, DS = [], {}, {}, {}
    for ds in a.datasets:
        day_cfg, lodo_cfg, M, P = best_configs(root, ds)
        cfg_rows += config_selection(root, ds, M, P)
        lc = dict(lodo_cfg, grpd='S_grpd'); dc = dict(day_cfg, grpd='S_grpd')
        # per-attack recall under LODO
        rec, LS[ds] = {}, {}
        for det in DETECTORS:
            LS[ds][det] = [lodo_scores(root, ds, det, s, lc) for s in SEEDS[det]]
            R = [recall_at(p) for p in LS[ds][det]]
            rec[det] = pd.concat([r['mean'] for r in R], axis=1).mean(axis=1); n = R[0]['size']
        T = pd.DataFrame(rec); T.insert(0, 'n', n)
        T.loc['types with recall >= 0.5'] = [np.nan] + [int((T[d] >= 0.5).sum()) for d in DETECTORS]
        T.round(3).to_csv(out / f'per_attack_{ds}.csv')
        DS[ds] = {det: [p[p.role == 'TEST'].reset_index(drop=True) for p in (day_scores(root, ds, det, s, dc) for s in DAY_SEEDS[det])]
                  for det in DETECTORS}
    pd.DataFrame(cfg_rows).round(4).to_csv(out / 'config_selection.csv', index=False)
    if 'cic2018' in LS:          # fold groups
        fg = {}
        for det, lst in LS['cic2018'].items():
            v = []
            for p in lst:
                sub = p.fold.isin(TEST_DAYS_2018)
                v.append([roc_auc_score(p[sub].y, p[sub].z), roc_auc_score(p[~sub].y, p[~sub].z), roc_auc_score(p.y, p.z)])
            fg[det] = dict(zip(['3_test_day_folds', '7_other_folds', 'all_10'], np.round(np.mean(v, 0), 4).tolist()))
        json.dump(fg, open(out / 'fold_groups.json', 'w'), indent=1)
        boot['2018_day_pcarec_minus_grpd'] = boot_diff(DS['cic2018']['pcarec'], DS['cic2018']['grpd'])
        boot['2018_lodo_grpd_minus_pcarec'] = boot_diff(LS['cic2018']['grpd'], LS['cic2018']['pcarec'])
    if 'cic2017' in LS:
        boot['2017_day_nflows_minus_grpd'] = boot_diff(DS['cic2017']['nflows'], DS['cic2017']['grpd'])
        boot['2017_lodo_pcarec_minus_grpd'] = boot_diff(LS['cic2017']['pcarec'], LS['cic2017']['grpd'])
        boot['2017_lodo_nflows_minus_grpd'] = boot_diff(LS['cic2017']['nflows'], LS['cic2017']['grpd'])
        # Attempted-only windows on the CIC-IDS2017 day-disjoint test
        _, _, M17, _ = best_configs(root, 'cic2017')
        dc = dict({f: M17[M17.group == f].sort_values('roc_auc_mean').score.iloc[-1] for f in FAMILIES if (M17.group == f).any()}, grpd='S_grpd')
        att = {}
        for det in DETECTORS:
            r = []
            for s in DAY_SEEDS[det]:
                p = day_scores(root, 'cic2017', det, s, dc); t = p[p.role == 'TEST']; q = p[p.role == 'TEST_ATTEMPTED']
                r.append([roc_auc_score(t.y, t.z), roc_auc_score(np.r_[t.y, np.ones(len(q))], np.r_[t.z, q.z]),
                          roc_auc_score(np.r_[t.y, np.zeros(len(q))], np.r_[t.z, q.z])])
            att[det] = dict(zip(['excluded', 'as_attack', 'as_benign'], np.round(np.mean(r, 0), 4).tolist()))
        json.dump(att, open(out / 'attempted_2017.json', 'w'), indent=1)
    json.dump(boot, open(out / 'paired_bootstrap.json', 'w'), indent=1)
    print('written to', out)


if __name__ == '__main__':
    main()
