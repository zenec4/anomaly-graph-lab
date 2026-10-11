"""Published-method baselines on the SAME window graphs, split and threshold rule (edge-level scores -> window scores).

anomale : Anomal-E (Caville et al., Knowledge-Based Systems 258, 2022). One E-GraphSAGE layer (constant-one node
          features, mean aggregation of [h_src || e_uv] messages, ReLU, 128-d node output), edge embedding
          z_uv = [z_u || z_v] (256-d), trained with Deep Graph Infomax: corruption = rows of the edge-feature
          matrix shuffled across edges (structure kept), readout = sigmoid(mean edge embedding), bilinear
          discriminator, BCE loss, Adam lr 1e-3. Graphs are made bidirectional as in the original paper. One DGI step per
          FIT window per epoch; early stopping on the CAL windows' DGI loss (patience 5, max 30 epochs).
          Detectors on edge embeddings, fitted on FIT edges only (benign-only, the original paper's 0% contamination
          setting), with fixed mid-grid settings of the original grid: PCA (PyOD, 10 components), IF (100 trees),
          CBLOF (5 clusters), HBOS (10 bins); contamination 0.01 (does not affect scores).
pcarec  : encoder-free control in the spirit of GIDS-Lite (Zhao & Hassan, arXiv 2609.12263): PCA (12 components)
          reconstruction error of the edge vector [edge features || source node features || destination node
          features], fitted on FIT edges.
Edge scores are aggregated per window: max, q99, top10 mean, mean. Labels are never read.
Features are z-scored with FIT statistics (builder features are already log-transformed) and clipped to +-10.
Usage: python baselines_edge.py <data_dir> <split.csv> <out_dir> --method anomale|pcarec [--seed 42] [--threads 2]
       [--max-fit-edges 200000] [--device auto|cpu|cuda] [--reuse-dgi]
Writes <out_dir>/window_scores.parquet (graph_id + <METHOD>_<det>_<agg>), config.json, timing.json, DONE.
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


def stats(graphs, idx, key):
    n, s, ss = 0, None, None
    for i in idx:
        v = graphs[i][key].double()
        s = v.sum(0) if s is None else s + v.sum(0); ss = (v ** 2).sum(0) if ss is None else ss + (v ** 2).sum(0); n += v.shape[0]
    mu = s / n; sd = torch.sqrt(torch.clamp(ss / n - mu ** 2, min=1e-12)).clamp_min(1e-6)
    return mu.float(), sd.float()


def zs(t, mu, sd):
    return torch.nan_to_num((t.float() - mu) / sd, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)


class EGraphSAGE(nn.Module):
    """One E-GraphSAGE layer as used by Anomal-E."""
    def __init__(self, ndim, edim, out=128):
        super().__init__()
        self.msg = nn.Linear(ndim + edim, out)
        self.apply_ = nn.Linear(ndim + out, out)

    def forward(self, h, ei, e):
        src, dst = ei
        m = self.msg(torch.cat([h[src], e], 1))
        agg = torch.zeros(h.size(0), m.size(1), device=h.device).index_add_(0, dst, m)
        cnt = torch.zeros(h.size(0), 1, device=h.device).index_add_(0, dst, torch.ones(dst.size(0), 1, device=h.device)).clamp_min(1)
        z = F.relu(self.apply_(torch.cat([h, agg / cnt], 1)))
        return torch.cat([z[src], z[dst]], 1)          # edge embeddings z_uv = [z_u || z_v]


class DGI(nn.Module):
    def __init__(self, edim, out=128):
        super().__init__()
        self.enc = EGraphSAGE(edim, edim, out)
        self.W = nn.Bilinear(2 * out, 2 * out, 1)

    def loss(self, ei, e):
        h = torch.ones(int(ei.max()) + 1, e.size(1), device=e.device)
        pos = self.enc(h, ei, e)
        neg = self.enc(h, ei, e[torch.randperm(e.size(0), device=e.device)])
        s = torch.sigmoid(pos.mean(0, keepdim=True)).expand_as(pos)
        lp, ln = self.W(pos, s).squeeze(1), self.W(neg, s).squeeze(1)
        return F.binary_cross_entropy_with_logits(torch.cat([lp, ln]), torch.cat([torch.ones_like(lp), torch.zeros_like(ln)]))

    @torch.no_grad()
    def embed(self, ei, e):
        return self.enc(torch.ones(int(ei.max()) + 1, e.size(1), device=e.device), ei, e)


DEV = torch.device('cpu')


def bidir(g, mu, sd):
    ei = g['edge_index'].long(); e = zs(g['edge_attr'], mu, sd)
    return torch.cat([ei, ei.flip(0)], 1).to(DEV), torch.cat([e, e], 0).to(DEV), ei.shape[1]


def aggregate(scores_by_graph, prefix):
    rows = []
    for gid, v in scores_by_graph.items():
        r = dict(graph_id=gid)
        for name, s in v.items():
            d = np.sort(s)[::-1]
            r[f'{prefix}_{name}_max'] = float(d[0]); r[f'{prefix}_{name}_q99'] = float(np.quantile(s, 0.99))
            r[f'{prefix}_{name}_top10'] = float(d[:10].mean()); r[f'{prefix}_{name}_mean'] = float(s.mean())
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('split'); ap.add_argument('out')
    ap.add_argument('--method', choices=['anomale', 'pcarec'], required=True)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--max-fit-edges', type=int, default=200_000); ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--reuse-dgi', action='store_true', help='reuse <out>/dgi.pt from an interrupted run (no retraining)')
    ap.add_argument('--device', default='auto', help='auto | cpu | cuda (DGI training/embedding only; detectors run on CPU)')
    a = ap.parse_args()
    global DEV
    DEV = torch.device('cuda' if (a.device == 'auto' and torch.cuda.is_available()) else ('cpu' if a.device == 'auto' else a.device))
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(dict(vars(a), split=str(a.split), device_used=str(DEV)), indent=1))
    t0 = time.time(); T = {}
    graphs, role = load(a.data, a.split)
    fit = np.flatnonzero(role == 'FIT'); cal = np.flatnonzero(role == 'CAL')
    if not len(cal):
        cal = np.flatnonzero(role == 'VAL')
    score_ids = np.flatnonzero(np.isin(role, ['VAL', 'TEST', 'TEST_NEAR', 'TEST_ATTEMPTED']))
    emu, esd = stats(graphs, fit, 'edge_attr')
    rng = np.random.default_rng(a.seed)
    per = max(1, a.max_fit_edges // max(len(fit), 1))

    if a.method == 'anomale':
        from pyod.models.pca import PCA as PPCA
        from pyod.models.iforest import IForest
        from pyod.models.cblof import CBLOF
        from pyod.models.hbos import HBOS
        model = DGI(graphs[0]['edge_attr'].shape[1]).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, bad, hist = np.inf, 0, []
        for ep in range(1, 0 if (a.reuse_dgi and (out / 'dgi.pt').exists()) else a.max_epochs + 1):
            model.train(); tl = 0.0
            for i in rng.permutation(fit):
                ei, e, _ = bidir(graphs[i], emu, esd)
                opt.zero_grad(); l = model.loss(ei, e); l.backward(); opt.step(); tl += float(l)
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([float(model.loss(*bidir(graphs[i], emu, esd)[:2])) for i in cal]))
            hist.append(dict(epoch=ep, train=tl / len(fit), cal=vl)); print(f'epoch {ep} train {tl / len(fit):.4f} cal {vl:.4f}', flush=True)
            if vl < best - 1e-4:
                best, bad = vl, 0; torch.save(model.state_dict(), out / 'dgi.pt')
            else:
                bad += 1
                if bad >= 5:
                    break
        model.load_state_dict(torch.load(out / 'dgi.pt', map_location=DEV)); model.eval()
        if hist:
            pd.DataFrame(hist).to_csv(out / 'train_history.csv', index=False)
        T['train_s'] = time.time() - t0

        def emb(i):
            ei, e, m = bidir(graphs[i], emu, esd)
            return model.embed(ei, e)[:m].cpu().numpy()           # one embedding per original (directed) edge
        Xf = np.vstack([emb(i)[rng.permutation(graphs[i]['edge_index'].shape[1])[:per]] for i in fit])
        dets = dict(PCA=PPCA(n_components=10, contamination=0.01, random_state=a.seed),
                    IF=IForest(n_estimators=100, contamination=0.01, random_state=a.seed, n_jobs=a.threads),
                    CBLOF=CBLOF(n_clusters=5, contamination=0.01, random_state=a.seed),
                    HBOS=HBOS(n_bins=10, contamination=0.01))
        for n in list(dets):
            try:
                dets[n].fit(Xf)
            except ValueError:          # CBLOF: no valid large/small cluster separation -> relax alpha/beta
                for al, be in ((0.9, 3), (0.95, 2), (0.8, 2)):
                    try:
                        dets[n] = CBLOF(n_clusters=5, alpha=al, beta=be, contamination=0.01, random_state=a.seed).fit(Xf)
                        T[f'{n}_alpha_beta'] = [al, be]; break
                    except ValueError:
                        continue
                else:
                    raise
        T['detectors_fit_s'] = time.time() - t0 - T['train_s']; T['fit_edges'] = len(Xf)
        S = {}
        for i in score_ids:
            z = emb(i); S[int(i)] = {n: d.decision_function(z) for n, d in dets.items()}
            if len(S) % 100 == 0:
                print(f'scored {len(S)}/{len(score_ids)} windows', flush=True)
        W = aggregate(S, 'ANOMALE')
    else:
        from sklearn.decomposition import PCA
        xmu, xsd = stats(graphs, fit, 'x')

        def vec(i):
            g = graphs[i]; s, d = g['edge_index'].long()
            x = zs(g['x'], xmu, xsd)
            return torch.cat([zs(g['edge_attr'], emu, esd), x[s], x[d]], 1).numpy()
        Xf = np.vstack([vec(i)[rng.permutation(graphs[i]['edge_index'].shape[1])[:per]] for i in fit])
        p = PCA(n_components=12, random_state=a.seed).fit(Xf)
        T['fit_edges'] = len(Xf)
        S = {}
        for i in score_ids:
            v = vec(i); S[int(i)] = dict(REC=((v - p.inverse_transform(p.transform(v))) ** 2).mean(1))
        W = aggregate(S, 'PCAREC')
    T['total_s'] = time.time() - t0
    W.to_parquet(out / 'window_scores.parquet', index=False)
    (out / 'timing.json').write_text(json.dumps(T, indent=1))
    (out / 'DONE').write_text(json.dumps(dict(windows=len(W), **T)))
    print('DONE', T, flush=True)


if __name__ == '__main__':
    main()
