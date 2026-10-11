"""Evaluate every run of one dataset (the only script that reads labels).

Usage: python evaluate.py <data_dir> <runs_dir> <out_dir> [--main grpd_s42] [--boot 2000]
<runs_dir> contains one sub-folder per run (written by grpd.py / baselines_graph.py / baselines_flow.py).
For every score: threshold = Q85 of its VAL windows (strict >); TEST = all strict-attack + safe benign windows of
the test days (role TEST in the run's split).
Outputs in <out_dir>:
  metrics.csv            one row per (run, score): Macro-F1 [block-bootstrap 95% CI], F1/recall/FPR, ROC-AUC [CI], AP,
                         TPR at FPR 1/5/10%, hour-matched AUC, volume-stratified and volume-adjusted AUC,
                         AUC/FPR with near-attack benign windows added (TEST + TEST_NEAR)
  metrics_by_seed.csv    mean / std / min / max over seeds for runs named <name>_s<seed>
  recall_by_type.csv     recall per attack type (TEST and TRAINDAY_ATTACK windows)
  fpr_by_day.csv         FPR per test day
  delong.csv             paired DeLong test of the main score vs every other score (TEST windows)
  weight_sensitivity.csv GRPD with all 13 fixed weights randomly perturbed (Dirichlet, 500 draws): AUC / Macro-F1 spread
  predictions.parquet    per-window labels + every score (for figures)
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

GRPD_COMPONENTS = ['S_grpd', 'S_residual_entity_gate', 'S_local_subgraph', 'S_local_profile_graph', 'S_local_if_graph', 'gae_loss']


def tpr_at(y, s, f):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= f + 1e-12].max())


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def block_bootstrap(y, s, thr, blocks, n, seed=7):
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks); idx_by = {b: np.flatnonzero(blocks == b) for b in ub}
    f1s, aucs = [], []
    for _ in range(n):
        i = np.concatenate([idx_by[b] for b in rng.choice(ub, len(ub), replace=True)])
        if 0 < y[i].sum() < len(i):
            f1s.append(f1_score(y[i], (s[i] > thr).astype(int), average='macro')); aucs.append(roc_auc_score(y[i], s[i]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return q(f1s), q(aucs)


def delong(y, s1, s2):
    """Paired DeLong test for two correlated ROC AUCs. Returns (auc1, auc2, p)."""
    pos, neg = y == 1, y == 0
    def comps(s):
        X, Y = s[pos], s[neg]
        V10 = np.array([(np.sum(x > Y) + 0.5 * np.sum(x == Y)) / len(Y) for x in X])
        V01 = np.array([(np.sum(X > v) + 0.5 * np.sum(X == v)) / len(X) for v in Y])
        return V10.mean(), V10, V01
    a1, v10a, v01a = comps(s1); a2, v10b, v01b = comps(s2)
    S10 = np.cov(np.vstack([v10a, v10b])); S01 = np.cov(np.vstack([v01a, v01b]))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / len(v10a) + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / len(v01a)
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return a1, a2, float(2 * st.norm.sf(abs(z)))


def evaluate(name, col, s_all, D, role, n_boot):
    val = role == 'VAL'; test = role == 'TEST'; near = role == 'TEST_NEAR'
    thr = float(np.quantile(s_all[val], 0.85))
    y = D.y.to_numpy()[test]; s = s_all[test]; p = (s > thr).astype(int)
    hours = D.hour.to_numpy()[test]; att_hours = set(hours[y == 1]); hm = np.isin(hours, list(att_hours))
    blocks = (D.source_file.astype(str) + '_' + D.hour.astype(str)).to_numpy()[test]
    ci_f, ci_a = block_bootstrap(y, s, thr, blocks, n_boot) if n_boot else ([np.nan] * 2, [np.nan] * 2)
    # volume control
    lv = np.log1p(D.n_flows_raw.to_numpy(dtype=float))
    lt = lv[test]; q = np.quantile(lt, [0.2, 0.4, 0.6, 0.8]); strata = np.digitize(lt, q)
    num = den = 0.0
    for k in np.unique(strata):
        m = strata == k
        if 0 < y[m].sum() < m.sum():
            w = y[m].sum() * (m.sum() - y[m].sum()); num += w * roc_auc_score(y[m], s[m]); den += w
    beta = np.polyfit(lv[val], s_all[val], 1) if val.sum() > 2 else np.array([0.0, 0.0])
    resid = s - np.polyval(beta, lt)
    # benign windows close to attacks added
    near = near & np.isfinite(s_all)          # runs with --timeline roles do not score near-attack windows
    tn = test | near
    y2 = D.y.to_numpy()[tn]; s2 = s_all[tn]
    return dict(run=name, score=col, threshold=thr, n_test=int(test.sum()), n_attack=int(y.sum()), n_benign=int((y == 0).sum()),
                macro_f1=f1_score(y, p, average='macro'), macro_f1_ci=ci_f, f1_attack=f1_score(y, p, zero_division=0),
                recall=float(p[y == 1].mean()), fpr=float(p[y == 0].mean()), precision=float(y[p == 1].mean()) if p.sum() else np.nan,
                roc_auc=safe_auc(y, s), roc_auc_ci=ci_a, ap=float(average_precision_score(y, s)),
                tpr_fpr1=tpr_at(y, s, .01), tpr_fpr5=tpr_at(y, s, .05), tpr_fpr10=tpr_at(y, s, .10),
                auc_hour_matched=safe_auc(y[hm], s[hm]), n_benign_hour_matched=int((y[hm] == 0).sum()),
                auc_volume_stratified=num / den if den else np.nan, auc_volume_adjusted=safe_auc(y, resid),
                auc_with_near_benign=safe_auc(y2, s2), fpr_near_benign=float((s_all[near] > thr).mean()) if near.any() else np.nan,
                n_near_benign=int(near.sum()))


def load_runs(runs_dir, data_dir):
    runs = {}
    for d in sorted(Path(runs_dir).iterdir()):
        f = d / 'window_scores.parquet'
        if not (d / 'DONE').exists() or not f.exists():
            continue
        W = pd.read_parquet(f)
        cfg = json.loads((d / 'config.json').read_text()) if (d / 'config.json').exists() else {}
        split = Path(cfg['split']).name if 'split' in cfg else 'split_day.csv'
        cols = [c for c in GRPD_COMPONENTS if c in W] if 'S_grpd' in W else [c for c in W.columns if c not in ('graph_id', 'n_flows_scored')]
        runs[d.name] = dict(W=W, cols=cols, split=split)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('runs'); ap.add_argument('out')
    ap.add_argument('--main', default='grpd_s42'); ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    L = pd.read_parquet(data / 'labels.parquet'); M = pd.read_parquet(data / 'meta.parquet')
    D = M[['graph_id', 'source_file', 'win_start', 'n_flows_raw']].merge(L[['graph_id', 'has_attack_strict', 'attack_type']], on='graph_id')
    D['win_start'] = pd.to_datetime(D.win_start); D['hour'] = D.win_start.dt.hour; D['y'] = D.has_attack_strict.astype(int)
    splits = {p.name: pd.read_csv(p).set_index('graph_id').role.reindex(D.graph_id).fillna('OTHER').to_numpy() for p in data.glob('split_*.csv')}
    runs = load_runs(a.runs, data)
    runs['confound'] = dict(W=D[['graph_id']].assign(n_flows=D.n_flows_raw.astype(float)), cols=['n_flows'], split='split_day.csv')
    rows, types, days, P = [], [], [], D[['graph_id', 'source_file', 'win_start', 'hour', 'n_flows_raw', 'y', 'attack_type']].copy()
    P['role_day'] = splits['split_day.csv']
    for name, r in runs.items():
        W = D[['graph_id']].merge(r['W'], on='graph_id', how='left')
        role = splits[r['split']]
        for c in r['cols']:
            s = W[c].to_numpy(dtype=float)
            ok = np.isin(role, ['VAL', 'TEST'])
            if np.isnan(s[ok]).any():
                print(f'skip {name}:{c} (missing scores)'); continue
            rows.append(evaluate(name, c, s, D, role, a.boot if (c in ('S_grpd',) or not name.startswith('grpd')) else 0))
            thr = rows[-1]['threshold']
            if r['split'] == 'split_day.csv':
                P[f'{name}:{c}'] = s
                det = s > thr
                for rl in ('TEST', 'TRAINDAY_ATTACK'):
                    m = (role == rl) & (D.y.to_numpy() == 1)
                    for t, g in pd.DataFrame(dict(t=D.attack_type[m], d=det[m])).groupby('t'):
                        types.append(dict(run=name, score=c, role=rl, attack_type=t, n=len(g), recall=g.d.mean()))
                m = (role == 'TEST') & (D.y.to_numpy() == 0)
                for dd, g in pd.DataFrame(dict(day=D.source_file[m], d=det[m])).groupby('day'):
                    days.append(dict(run=name, score=c, day=dd, n_benign=len(g), fpr=g.d.mean()))
        print(f'evaluated {name}', flush=True)
    R = pd.DataFrame(rows); R.to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(types).to_csv(out / 'recall_by_type.csv', index=False)
    pd.DataFrame(days).to_csv(out / 'fpr_by_day.csv', index=False)
    P.to_parquet(out / 'predictions.parquet', index=False)

    # seeds
    R['group'] = R.run.str.replace(r'_s\d+$', '', regex=True)
    num = ['macro_f1', 'roc_auc', 'ap', 'recall', 'fpr', 'tpr_fpr1', 'tpr_fpr5', 'tpr_fpr10', 'auc_hour_matched',
           'auc_volume_stratified', 'auc_volume_adjusted', 'auc_with_near_benign']
    G = R.groupby(['group', 'score'])[num].agg(['mean', 'std', 'min', 'max', 'count'])
    G.columns = ['_'.join(c) for c in G.columns]; G.reset_index().to_csv(out / 'metrics_by_seed.csv', index=False)

    # DeLong: main GRPD vs every other score on the day split
    role = splits['split_day.csv']; test = role == 'TEST'; y = D.y.to_numpy()[test]
    key = f'{a.main}:S_grpd'
    if key in P:
        dl = []
        for c in P.columns:
            if ':' in c and c != key:
                s1, s2 = P[key].to_numpy()[test], P[c].to_numpy(dtype=float)[test]
                if not np.isnan(s2).any():
                    a1, a2, pv = delong(y, s1, s2); dl.append(dict(other=c, auc_main=a1, auc_other=a2, p_value=pv))
        pd.DataFrame(dl).to_csv(out / 'delong.csv', index=False)

    # weight sensitivity of the GRPD fusion (window-level channels saved by grpd.py)
    f = Path(a.runs) / a.main / 'window_scores.parquet'
    if f.exists():
        W = D[['graph_id']].merge(pd.read_parquet(f), on='graph_id', how='left')
        val = role == 'VAL'
        rng = np.random.default_rng(0); res = []
        def comb(w):
            fus = (w['a'][0] * W.S_node_residual_profile + w['a'][1] * W.S_edge_residual_profile + w['a'][2] * W.S_node_local_hi
                   + w['a'][3] * W.S_residual_concentration)
            gate = fus * ((1 - w['g']) + w['g'] * W.E_res_entity_persistence)
            sub = (w['b'][0] * W.S_local_profile_graph + w['b'][1] * W.S_local_if_graph + w['b'][2] * W.S_local_temporal_graph
                   + w['b'][3] * W.S_local_rescue_graph)
            return (w['c'][0] * gate + w['c'][1] * sub).to_numpy(dtype=float)
        base = dict(a=np.array([.6, .22, .1, .08]), g=0.3, b=np.array([.42, .24, .2, .14]), c=np.array([.62, .38]))
        cj = Path(a.runs) / a.main / 'config.json'
        cc = json.loads(cj.read_text()).get('cfg') if cj.exists() else None
        if cc:   # centre the perturbations on the weights the run actually used (from config.json)
            base = dict(a=np.array(cc['res_w']), g=cc['gate_g'], b=np.array(cc['local_w']), c=np.array(cc['final_w']))
        for i in range(501):
            if i == 0:
                w = base
            else:
                w = dict(a=rng.dirichlet(base['a'] * 20), g=float(np.clip(rng.normal(base['g'], 0.1), 0, 1)),
                         b=rng.dirichlet(base['b'] * 20), c=rng.dirichlet(base['c'] * 20))
            s = comb(w); thr = np.quantile(s[val], 0.85)
            res.append(dict(draw=i, roc_auc=roc_auc_score(y, s[test]), macro_f1=f1_score(y, (s[test] > thr).astype(int), average='macro'),
                            tpr_fpr5=tpr_at(y, s[test], .05)))
        S = pd.DataFrame(res); S.to_csv(out / 'weight_sensitivity.csv', index=False)
        print('weight sensitivity (AUC 5/50/95%):', np.percentile(S.roc_auc[1:], [5, 50, 95]).round(3), 'base', round(S.roc_auc[0], 3))
    show = ['run', 'score', 'macro_f1', 'macro_f1_ci', 'roc_auc', 'ap', 'tpr_fpr1', 'tpr_fpr5', 'auc_hour_matched', 'auc_volume_adjusted', 'fpr']
    print(R.loc[R.score.isin(['S_grpd', 'n_flows']) | ~R.run.str.startswith('grpd'), show].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
