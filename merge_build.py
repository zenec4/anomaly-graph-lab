"""Merge the per-day builds of one dataset into the model input folder and write the splits.

Usage: python merge_build.py <dataset> <days_dir> <out_dir> [window_minutes=5]      (dataset: cic2018 | cic2017; see datasets.py)
Writes to <out_dir>:
  graphs.pt            list of dicts (x, edge_index, edge_attr, graph_attr), position == graph_id
  meta.parquet         label-free window table: graph_id, source_file, win_start, win_end, n_nodes, n_edges, n_flows_raw
  labels.parquet       labels, read only by evaluate.py: has_attack_strict, has_attack_any, raw_attempted_flows, attack_type, ...
  node_ips.pkl         per window: list of node IPs
  feature_schema.json
  split_day.csv        graph_id, role  (FIT / CAL / VAL / TEST / TEST_NEAR / TEST_ATTEMPTED / TRAINDAY_ATTACK / OTHER)
  split_random.csv     same roles drawn at random within days (demonstration of the random-split pitfall)
  split_summary.json   counts per role / day
Role rules:
  safe benign = window without any non-BENIGN flow and without any non-pure window (attack or Attempted) within
                +-3 windows (15 min) of the same day.
  FIT/CAL/VAL days -> safe benign windows only; TEST days -> every strict-attack window + safe benign windows.
  TEST_NEAR: pure benign windows of TEST days that fail the safe gap (scored, reported separately).
  TEST_ATTEMPTED: windows of TEST days whose only non-benign flows are 'Attempted' (excluded from metrics).
  TRAINDAY_ATTACK: strict-attack windows on FIT/CAL/VAL days (secondary: recall per attack type).
"""
import json, pickle, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import DATASETS

SAFE_GAP_MIN = 15          # benign windows within +-15 min of any non-pure window are not 'safe'
WINDOW_MIN = 5


def safe_benign(meta):
    pure = meta.has_attack_any.eq(0) & meta.raw_attempted_flows.eq(0)
    bad = {(s, t) for s, t in zip(meta.source_file[~pure], meta.win_start[~pure])}
    step = pd.Timedelta(minutes=WINDOW_MIN); gap = -(-SAFE_GAP_MIN // WINDOW_MIN)
    safe = [bool(p) and all((s, t + k * step) not in bad for k in range(-gap, gap + 1))
            for p, s, t in zip(pure, meta.source_file, meta.win_start)]
    return pure.to_numpy(), np.asarray(safe)


def day_roles(meta, cfg):
    pure, safe = safe_benign(meta)
    strict = meta.has_attack_strict.eq(1).to_numpy()
    day_role = meta.source_file.map({d: r for r, ds in cfg['roles'].items() for d in ds}).to_numpy()
    role = np.full(len(meta), 'OTHER', dtype=object)
    train_day = np.isin(day_role, ['FIT', 'CAL', 'VAL'])
    for r in ('FIT', 'CAL', 'VAL'):
        role[(day_role == r) & safe] = r
    role[train_day & strict] = 'TRAINDAY_ATTACK'
    test = day_role == 'TEST'
    role[test & (strict | safe)] = 'TEST'
    role[test & pure & ~safe] = 'TEST_NEAR'
    role[test & ~pure & ~strict] = 'TEST_ATTEMPTED'
    # tail_of_fit: e.g. [['CAL', 0.15]] -> the last 15% (by time) of each FIT day's safe benign windows become CAL
    for r, frac in cfg.get('tail_of_fit', []):
        for d in cfg['roles']['FIT']:
            idx = np.flatnonzero((meta.source_file.to_numpy() == d) & (role == 'FIT'))
            idx = idx[np.argsort(meta.win_start.to_numpy()[idx], kind='mergesort')]
            k = int(round(frac * len(idx)))
            if k:
                role[idx[len(idx) - k:]] = r
    return role


def random_roles(meta, day_role, seed=42):
    """Same role sizes, but safe benign windows of ALL days shuffled into FIT/CAL/VAL/TEST; all attacks in TEST."""
    pure, safe = safe_benign(meta)
    strict = meta.has_attack_strict.eq(1).to_numpy()
    role = np.full(len(meta), 'OTHER', dtype=object)
    sizes = {r: int(((day_role == r) & safe).sum()) for r in ('FIT', 'CAL', 'VAL', 'TEST')}
    pool = np.random.default_rng(seed).permutation(np.flatnonzero(safe))
    a = 0
    for r in ('FIT', 'CAL', 'VAL', 'TEST'):
        role[pool[a:a + sizes[r]]] = r; a += sizes[r]
    role[pool[a:]] = 'TEST'
    role[strict] = 'TEST'
    return role


def main():
    global WINDOW_MIN
    name, days_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if len(sys.argv) > 4:
        WINDOW_MIN = int(sys.argv[4])
    cfg = DATASETS[name]; out.mkdir(parents=True, exist_ok=True)
    graphs, metas, ips = [], [], []
    for d in cfg['days']:
        stem = Path(d).stem
        assert (days_dir / f'{stem}.done').exists(), f'day not built: {stem}'
        graphs += torch.load(days_dir / f'{stem}.pt', weights_only=False)
        m = pd.read_parquet(days_dir / f'{stem}_meta.parquet')
        metas.append(m)
        with open(days_dir / f'{stem}_ips.pkl', 'rb') as f:
            ips += pickle.load(f)
    meta = pd.concat(metas, ignore_index=True)
    meta['graph_id'] = np.arange(len(meta))
    meta['win_start'] = pd.to_datetime(meta['win_start'])
    assert len(graphs) == len(meta) == len(ips)
    assert not meta.duplicated(['source_file', 'win_start']).any()
    torch.save(graphs, out / 'graphs.pt')
    meta[['graph_id', 'source_file', 'win_start', 'win_end', 'n_nodes', 'n_edges', 'n_flows_raw']].to_parquet(out / 'meta.parquet', index=False)
    meta[['graph_id', 'source_file', 'win_start', 'has_attack_strict', 'has_attack_any', 'raw_any_nonbenign_flows',
          'raw_strict_attack_flows', 'raw_attempted_flows', 'attack_type', 'n_flows_raw']].to_parquet(out / 'labels.parquet', index=False)
    with open(out / 'node_ips.pkl', 'wb') as f:
        pickle.dump(ips, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy(days_dir / 'feature_schema.json', out / 'feature_schema.json')

    role = day_roles(meta, cfg)
    pd.DataFrame(dict(graph_id=meta.graph_id, role=role)).to_csv(out / 'split_day.csv', index=False)
    day_role = meta.source_file.map({d: r for r, ds in cfg['roles'].items() for d in ds}).to_numpy()
    for r, _ in cfg.get('tail_of_fit', []):
        day_role = np.where(role == r, r, day_role)
    rrole = random_roles(meta, day_role)
    pd.DataFrame(dict(graph_id=meta.graph_id, role=rrole)).to_csv(out / 'split_random.csv', index=False)

    t = pd.DataFrame(dict(day=meta.source_file, role=role, attack=meta.has_attack_strict))
    summary = dict(dataset=name, windows=len(meta), attack_windows=int(meta.has_attack_strict.sum()),
                   day_split=t.groupby(['day', 'role']).size().unstack(fill_value=0).to_dict(orient='index'),
                   day_split_roles=pd.Series(role).value_counts().to_dict(),
                   random_split_roles=pd.Series(rrole).value_counts().to_dict(),
                   test_attack_types=meta.loc[(role == 'TEST') & meta.has_attack_strict.eq(1).to_numpy(), 'attack_type'].value_counts().to_dict(),
                   trainday_attack_types=meta.loc[role == 'TRAINDAY_ATTACK', 'attack_type'].value_counts().to_dict(),
                   nodes_total=int(meta.n_nodes.sum()), edges_total=int(meta.n_edges.sum()))
    (out / 'split_summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == '__main__':
    main()
