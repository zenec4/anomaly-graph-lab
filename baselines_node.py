"""Node-level graph anomaly detectors on the SAME window graphs, split and threshold rule (node scores -> window scores).

Both detectors are trained inductively on all FIT windows (benign only, one window = one graph per step), use the 30 node
features (z-scored with FIT statistics, clipped to +-10) and the undirected window topology with self-loops.

dominant : DOMINANT (Ding et al., SDM 2019). GCN encoder (3 layers, 64 units, ReLU), attribute decoder (2 GCN layers)
           and structure decoder sigmoid(z_i . z_j). Node score = 0.5 * attribute error + 0.5 * structure error, as in the
           original paper and PyGOD's default weight. Because dense adjacency rows are infeasible for windows with tens of
           thousands of nodes, the structure error of node i is computed on its edges and on the same number of random
           non-neighbours of the window (root of the mean squared error over these pairs).
ocgnn    : OCGNN (Wang et al., Neural Computing and Applications 2021). GCN encoder without bias (3 layers, 64 units),
           Deep-SVDD objective: centre c = mean embedding of the FIT nodes after initialisation (|c_k| >= 0.1),
           loss = mean ||z - c||^2 (one-class, soft-boundary term not used); node score = ||z - c||^2.
Training: Adam lr 4e-3 (PyGOD default), at most 30 epochs, early stopping on the CAL windows (patience 5).
Window score per detector: max, q99, top-10 mean and mean of node scores. Labels are never read.
Usage: python baselines_node.py <data_dir> <split.csv> <out_dir> --method dominant|ocgnn [--seed 42] [--threads 2] [--device auto]
Writes <out_dir>/window_scores.parquet (graph_id + <METHOD>_<agg>), config.json, timing.json, DONE.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def load(data, split):
    graphs = torch.load(Path(data) / 'graphs.pt', map_location='cpu', weights_only=False, mmap=True)
    role = pd.read_csv(split).set_index('graph_id').role.reindex(range(len(graphs))).fillna('OTHER').to_numpy()
    return graphs, role


def node_stats(graphs, idx):
    n, s, ss = 0, None, None
    for i in idx:
        v = graphs[i]['x'].double()
        s = v.sum(0) if s is None else s + v.sum(0); ss = (v ** 2).sum(0) if ss is None else ss + (v ** 2).sum(0); n += v.shape[0]
    mu = s / n; sd = torch.sqrt(torch.clamp(ss / n - mu ** 2, min=1e-12)).clamp_min(1e-6)
    return mu.float(), sd.float()


def gcn_norm(ei, n, dev):
    """Symmetric normalised adjacency with self-loops of an undirected graph, as a sparse tensor."""
    ei = torch.cat([ei, ei.flip(0), torch.arange(n).repeat(2, 1)], 1)
    ei = torch.unique(ei, dim=1)
    deg = torch.zeros(n).index_add_(0, ei[0], torch.ones(ei.shape[1]))
    w = deg[ei[0]].rsqrt() * deg[ei[1]].rsqrt()
    return torch.sparse_coo_tensor(ei, w, (n, n)).coalesce().to(dev)


class GCN(nn.Module):
    def __init__(self, dims, bias=True, last_act=False):
        super().__init__()
        self.lins = nn.ModuleList(nn.Linear(a, b, bias=bias) for a, b in zip(dims[:-1], dims[1:]))
        self.last_act = last_act

    def forward(self, x, A):
        for k, lin in enumerate(self.lins):
            x = torch.sparse.mm(A, lin(x))
            if k < len(self.lins) - 1 or self.last_act:
                x = F.relu(x)
        return x


class Dominant(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.enc = GCN([d, h, h, h], last_act=True)
        self.att = GCN([h, h, d])

    def node_scores(self, x, A, ei, rng_gen):
        z = self.enc(x, A); xh = self.att(z, A)
        ea = (((x - xh) ** 2).sum(1) + 1e-12).sqrt()
        n = x.shape[0]
        und = torch.cat([ei, ei.flip(0)], 1)
        neg_dst = torch.randint(0, n, (und.shape[1],), device=x.device, generator=rng_gen)
        pos = torch.sigmoid((z[und[0]] * z[und[1]]).sum(1)); neg = torch.sigmoid((z[und[0]] * z[neg_dst]).sum(1))
        err = torch.zeros(n, device=x.device).index_add_(0, und[0], (1 - pos) ** 2 + neg ** 2)
        cnt = torch.zeros(n, device=x.device).index_add_(0, und[0], torch.full_like(pos, 2.0)).clamp_min(1)
        es = (err / cnt + 1e-12).sqrt()
        return 0.5 * ea + 0.5 * es


class OCGNN(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.enc = GCN([d, h, h, h], bias=False)
        self.register_buffer('c', torch.zeros(h))

    def node_scores(self, x, A, ei, rng_gen):
        return ((self.enc(x, A) - self.c) ** 2).sum(1)


def aggregate(S, prefix):
    rows = []
    for gid, s in S.items():
        d = np.sort(s)[::-1]
        rows.append({'graph_id': gid, f'{prefix}_max': float(d[0]), f'{prefix}_q99': float(np.quantile(s, 0.99)),
                     f'{prefix}_top10': float(d[:10].mean()), f'{prefix}_mean': float(s.mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('split'); ap.add_argument('out')
    ap.add_argument('--method', choices=['dominant', 'ocgnn'], required=True)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--max-epochs', type=int, default=30); ap.add_argument('--device', default='auto')
    a = ap.parse_args()
    dev = torch.device('cuda' if (a.device == 'auto' and torch.cuda.is_available()) else ('cpu' if a.device == 'auto' else a.device))
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(dict(vars(a), split=str(a.split), device_used=str(dev)), indent=1))
    t0 = time.time(); T = {}
    graphs, role = load(a.data, a.split)
    fit = np.flatnonzero(role == 'FIT'); cal = np.flatnonzero(role == 'CAL')
    if not len(cal):
        cal = np.flatnonzero(role == 'VAL')
    score_ids = np.flatnonzero(np.isin(role, ['VAL', 'TEST', 'TEST_NEAR', 'TEST_ATTEMPTED']))
    mu, sd = node_stats(graphs, fit)

    def prep(i):
        g = graphs[i]; x = torch.nan_to_num((g['x'].float() - mu) / sd, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
        ei = g['edge_index'].long()
        return x.to(dev), gcn_norm(ei, x.shape[0], dev), ei.to(dev)

    d = graphs[0]['x'].shape[1]
    model = (Dominant(d) if a.method == 'dominant' else OCGNN(d)).to(dev)
    if a.method == 'ocgnn':      # Deep-SVDD centre from the initialised encoder, as in OCGNN
        with torch.no_grad():
            c = torch.stack([model.enc(*prep(i)[:2]).mean(0) for i in fit]).mean(0)
            c[(c.abs() < 0.1) & (c < 0)] = -0.1; c[(c.abs() < 0.1) & (c >= 0)] = 0.1
            model.c.copy_(c)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    rng = np.random.default_rng(a.seed)
    best, bad, hist = np.inf, 0, []
    for ep in range(1, a.max_epochs + 1):
        model.train(); tl = 0.0
        for i in rng.permutation(fit):
            x, A, ei = prep(i)
            opt.zero_grad(); loss = model.node_scores(x, A, ei, gen).mean(); loss.backward(); opt.step(); tl += float(loss)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(model.node_scores(*prep(i), gen).mean()) for i in cal]))
        hist.append(dict(epoch=ep, train=tl / len(fit), cal=vl)); print(f'epoch {ep} train {tl / len(fit):.4f} cal {vl:.4f}', flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0; torch.save(model.state_dict(), out / 'model.pt')
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(torch.load(out / 'model.pt', map_location=dev)); model.eval()
    pd.DataFrame(hist).to_csv(out / 'train_history.csv', index=False)
    T['train_s'] = time.time() - t0
    S = {}
    with torch.no_grad():
        for i in score_ids:
            S[int(i)] = model.node_scores(*prep(i), gen).cpu().numpy()
    W = aggregate(S, a.method.upper())
    T['total_s'] = time.time() - t0
    W.to_parquet(out / 'window_scores.parquet', index=False)
    (out / 'timing.json').write_text(json.dumps(T, indent=1))
    (out / 'DONE').write_text(json.dumps(dict(windows=len(W), **T)))
    print('DONE', T, flush=True)


if __name__ == '__main__':
    main()
