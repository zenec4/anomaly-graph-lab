"""GRPD (graph residual profiling detector): one-class scoring of traffic windows (IP interaction graphs).

Pipeline (fitted on benign windows only; labels are never read here):
  1. Graph autoencoder (node, edge and window-attribute reconstruction; two GraphSAGE layers or, with --encoder mlp,
     per-node linear layers) trained on FIT windows, early stopping on CAL windows.
  2. Residual profiles: per-feature robust profile (median and IQR) of log-residuals on FIT; per-entity profile scores;
     window-level residual channels (ECDF against VAL windows), top-entity persistence over the previous 3 windows
     and the residual entity gate.
  3. Node-local detector: per-node residual, incident-edge residual, local traffic and causal per-IP history features;
     an isolation forest fitted on FIT nodes; node scores aggregated per window (ECDF against VAL windows).
  Final score S_grpd = (gate + local) / 2. All fusions use equal weights; optimiser, isolation forest and scalers use
  library defaults (see CFG).

Ablation switches: --encoder mlp (no message passing) | --graph-attr base (no window-history features)
                   --no-ip-history (no per-IP / top-entity persistence) | --no-edge-attr | --ae-from <run> (reuse an AE)

Usage:
  python grpd.py --data <graph_dir> --split <split.csv> --out <run_dir> [--seed 42] [--encoder sage|mlp] [--threads 16]
Outputs in <run_dir>: config.json, log.txt, ae/ (model and normalization statistics), profiles.npz,
  window_scores.parquet (graph_id + all channels incl. S_grpd), node_top.parquet (top nodes per window), DONE.
"""
from __future__ import annotations
import argparse, copy, json, os, pickle, random, sys, time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ fixed hyperparameters
AE = dict(hidden=128, epochs=80, patience=15, plateau_patience=5, plateau_factor=0.5)   # optimiser settings: see CFG
NODE_SAMPLE_PER_FIT_GRAPH, EDGE_SAMPLE_PER_FIT_GRAPH = 256, 512
PROFILE_Q_LOW, PROFILE_Q_HIGH, PROFILE_SCALE_EPS, PROFILE_Z_CLIP = 0.1, 0.9, 1e-4, 30.0
TOP_NODE_K = TOP_EDGE_K = 10
HISTORY_WINDOWS, MAX_GAP_MIN = 3, 7.5
MAX_FIT_NODES_PER_GRAPH, FIT_TAIL_NODES_PER_GRAPH = 300, 60
# Configuration: equal weights in every fusion; library defaults for the optimiser (Adam, lr 1e-3, no weight decay,
# dropout, clipping or LR schedule), the isolation forest (100 trees, max_samples='auto', all features) and robust
# scaling (IQR); uniform node sampling for the forest.
CFG = dict(name='default', profile_w=(0.2, 0.2, 0.2, 0.2, 0.2), persist_w=(0.5, 0.5), res_w=(0.25, 0.25, 0.25, 0.25), gate_g=0.5,
           node_w=(0.25, 0.25, 0.25, 0.25), temporal_g=0.5, rescue_w=(1.0, 1.0), local_w=(0.25, 0.25, 0.25, 0.25), final_w=(0.5, 0.5),
           profile_q=(0.25, 0.75), node_scaler_q=(25.0, 75.0), if_trees=100, if_max_samples='auto', if_max_features=1.0, tail_nodes=0,
           opt='adam', lr=1e-3, weight_decay=0.0, dropout=0.0, clip=None, plateau=False)
TOP_IPS = 10
N_GRAPH_BASE = 47          # first 47 entries of graph_attr = current-window features (rest = history deltas/z)

LOG = None


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if LOG is not None:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


# ------------------------------------------------------------------ data
def load_data(data_dir, graph_attr_mode, no_edge_attr=False):
    import torch
    d = Path(data_dir)
    meta = pd.read_parquet(d / 'meta.parquet')
    graphs = torch.load(d / 'graphs.pt', map_location='cpu', weights_only=False, mmap=True)
    with open(d / 'node_ips.pkl', 'rb') as f:
        node_ips = pickle.load(f)
    schema = json.loads((d / 'feature_schema.json').read_text())
    assert len(graphs) == len(meta) == len(node_ips)
    assert meta.graph_id.tolist() == list(range(len(meta))), 'meta must be ordered by graph_id = position'
    out = []
    for g in graphs:
        ga = g['graph_attr'].float().view(1, -1)
        if graph_attr_mode == 'base':
            ga = ga[:, :N_GRAPH_BASE]
        ea = g['edge_attr'].float()
        # --no-edge-attr: the autoencoder sees (and reconstructs) no edge attributes; raw ones still feed node-local traffic features
        out.append(dict(x=g['x'].float(), edge_index=g['edge_index'].long(), graph_attr=ga, edge_raw=ea,
                        edge_attr=torch.zeros(ea.shape[0], 1) if no_edge_attr else ea))
    del graphs
    meta['win_start'] = pd.to_datetime(meta['win_start'])
    return meta, out, node_ips, schema


# ------------------------------------------------------------------ model
def build_model(x_dim, edge_dim, graph_dim, encoder):
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

    def scatter_mean(src, index, dim_size):
        out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
        cnt = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        cnt.index_add_(0, index, torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype))
        return out / cnt.clamp_min(1.0)

    class MLP(nn.Module):
        def __init__(self, i, h, o, dropout):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o))

        def forward(self, v):
            return self.net(v)

    class GraphAE(nn.Module):
        """Single-graph forward of the graph autoencoder (same parameter names -> checkpoints compatible)."""
        def __init__(self, hidden=128, dropout=0.05):
            super().__init__()
            self.encoder = encoder
            self.node_in = MLP(x_dim, hidden, hidden, dropout)
            self.edge_in = MLP(edge_dim, hidden, hidden, dropout)
            self.graph_in = MLP(graph_dim, hidden, hidden, dropout)
            if encoder == 'sage':
                self.conv1, self.conv2 = SAGEConv(hidden, hidden), SAGEConv(hidden, hidden)
            else:   # ablation: same depth/width, no neighbour aggregation
                self.conv1, self.conv2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
            self.graph_fuse = MLP(hidden * 4, hidden, hidden, dropout)
            self.node_dec = MLP(hidden * 2, hidden, x_dim, dropout)
            self.edge_dec = MLP(hidden * 4, hidden, edge_dim, dropout)
            self.graph_dec = MLP(hidden, hidden, graph_dim, dropout)

        def forward(self, x, edge_index, edge_attr, graph_attr):
            h = self.node_in(x)
            he = self.edge_in(edge_attr)
            hg_in = self.graph_in(graph_attr)
            if self.encoder == 'sage':
                h = F.relu(self.conv1(h, edge_index)); h = F.relu(self.conv2(h, edge_index))
            else:
                h = F.relu(self.conv1(h)); h = F.relu(self.conv2(h))
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            # same pooling ops as the original batched implementation (bit-identical sums)
            hg = self.graph_fuse(torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch),
                                            scatter_mean(he, batch[edge_index[0]], 1), hg_in], dim=1))
            x_hat = self.node_dec(torch.cat([h, hg.expand(h.size(0), -1)], dim=1))
            s, t = edge_index
            e_hat = self.edge_dec(torch.cat([h[s], h[t], he, hg.expand(he.size(0), -1)], dim=1))
            return x_hat, e_hat, self.graph_dec(hg)

    return GraphAE(AE['hidden'], CFG['dropout'])


def normalization_stats(graphs):
    import torch
    def stream(key, dim):
        n, s, ss = 0, torch.zeros(dim, dtype=torch.float64), torch.zeros(dim, dtype=torch.float64)
        for g in graphs:
            v = g[key].view(-1, dim).double()
            n += v.size(0); s += v.sum(0); ss += (v ** 2).sum(0)
        mean = s / max(n, 1)
        std = torch.sqrt(torch.clamp(ss / max(n, 1) - mean ** 2, min=1e-12))
        return mean.float(), std.float().clamp_min(1e-6)
    x_mu, x_std = stream('x', graphs[0]['x'].shape[1])
    e_mu, e_std = stream('edge_attr', graphs[0]['edge_attr'].shape[1])
    g_mu, g_std = stream('graph_attr', graphs[0]['graph_attr'].shape[1])
    return dict(x_mu=x_mu, x_std=x_std, e_mu=e_mu, e_std=e_std, g_mu=g_mu, g_std=g_std)


def normalize(g, st, device):
    import torch
    nz = lambda t: torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
    return (nz((g['x'] - st['x_mu']) / st['x_std']).to(device), g['edge_index'].to(device),
            nz((g['edge_attr'] - st['e_mu']) / st['e_std']).to(device), nz((g['graph_attr'] - st['g_mu']) / st['g_std']).to(device))


def train_ae(graphs, fit_idx, cal_idx, out, encoder, seed, device):
    import torch, torch.nn.functional as F
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    stats = normalization_stats([graphs[i] for i in fit_idx])
    torch.save(stats, out / 'normalization_stats.pt')
    fit = [normalize(graphs[i], stats, device) for i in fit_idx]
    cal = [normalize(graphs[i], stats, device) for i in cal_idx]
    dims = (fit[0][0].shape[1], fit[0][2].shape[1], fit[0][3].shape[1])
    model = build_model(*dims, encoder).to(device)
    if CFG['opt'] == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=AE['plateau_factor'], patience=AE['plateau_patience']) if CFG['plateau'] else None
    loss_fn = lambda b: sum(F.mse_loss(p, t) for p, t in zip(model(*b), (b[0], b[2], b[3])))
    rng = np.random.default_rng(seed)
    best, best_epoch, bad, hist = float('inf'), -1, 0, []
    for epoch in range(1, AE['epochs'] + 1):
        model.train(); tl = 0.0
        for i in rng.permutation(len(fit)):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(fit[i]); loss.backward()
            if CFG['clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip'])
            opt.step()
            tl += float(loss.detach())
        model.eval()
        with torch.no_grad():
            vl = sum(float(loss_fn(b)) for b in cal) / max(len(cal), 1)
        tl /= max(len(fit), 1)
        if sched is not None:
            sched.step(vl)
        hist.append(dict(epoch=epoch, train_loss=tl, val_loss=vl, lr=opt.param_groups[0]['lr']))
        if vl < best - 1e-6:
            best, best_epoch, bad = vl, epoch, 0
            torch.save(dict(model_state=model.state_dict(), dims=dims, encoder=encoder, best_epoch=epoch, best_val=vl), out / 'model_best.pt')
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0 or bad == 0:
            log(f'AE epoch {epoch:03d} train={tl:.5f} cal={vl:.5f} best={best:.5f}@{best_epoch}')
        if bad >= AE['patience']:
            break
    pd.DataFrame(hist).to_csv(out / 'train_history.csv', index=False)
    (out / 'DONE').write_text(json.dumps(dict(best_epoch=best_epoch, best_val=best, epochs=len(hist))))


def load_ae(ae_dir, device):
    import torch
    ck = torch.load(Path(ae_dir) / 'model_best.pt', map_location='cpu', weights_only=False)
    stats = torch.load(Path(ae_dir) / 'normalization_stats.pt', map_location='cpu', weights_only=False)
    if 'dims' in ck:
        dims, enc = ck['dims'], ck.get('encoder', 'sage')
    else:   # checkpoint of an earlier version of the script
        c = ck['config']; dims, enc = (c['x_dim'], c['edge_dim'], c['graph_dim']), 'sage'
    model = build_model(*dims, enc).to(device)
    model.load_state_dict(ck['model_state'], strict=True); model.eval()
    return model, stats


# ------------------------------------------------------------------ helpers 
def sample_rows(values, maximum_rows, seed):
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= maximum_rows:
        return values
    return values[np.random.default_rng(seed).choice(len(values), size=maximum_rows, replace=False)]


def fit_profile_stats(samples):
    samples = np.asarray(samples, dtype=np.float32)
    median = np.quantile(samples, 0.5, axis=0).astype(np.float32)
    q_low = np.quantile(samples, CFG['profile_q'][0], axis=0).astype(np.float32)
    q_high = np.quantile(samples, CFG['profile_q'][1], axis=0).astype(np.float32)
    return dict(median=median, q_low=q_low, q_high=q_high, scale=np.maximum(q_high - q_low, PROFILE_SCALE_EPS).astype(np.float32))


def profile_components(log_residual, stats):
    z = (np.asarray(log_residual, dtype=np.float32) - stats['median'][None, :]) / stats['scale'][None, :]
    z = np.clip(z, 0.0, PROFILE_Z_CLIP)
    k = min(3, z.shape[1]); p = z.shape[1] - k
    top3 = np.partition(z, p, axis=1)[:, p:].mean(axis=1)
    mx, mean = z.max(axis=1), z.mean(axis=1)
    l2 = np.sqrt(np.mean(z ** 2, axis=1)); rare = (z > 3.0).mean(axis=1)
    w = CFG['profile_w']
    score = w[0] * top3 + w[1] * mx + w[2] * l2 + w[3] * mean + w[4] * rare
    f = lambda a: a.astype(np.float32)
    return dict(score=f(score), top3_mean=f(top3), max=f(mx), mean=f(mean), l2=f(l2), rare_fraction=f(rare))


def aggregate_vector(values, prefix):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return {f'{prefix}_{s}': 0.0 for s in ('mean', 'p90', 'p95', 'p99', 'max', 'top5', 'top10', 'top20')}
    d = np.sort(values)[::-1]
    tm = lambda k: float(d[:min(k, len(d))].mean())
    return {f'{prefix}_mean': float(values.mean()), f'{prefix}_p90': float(np.quantile(values, 0.9)),
            f'{prefix}_p95': float(np.quantile(values, 0.95)), f'{prefix}_p99': float(np.quantile(values, 0.99)),
            f'{prefix}_max': float(values.max()), f'{prefix}_top5': tm(5), f'{prefix}_top10': tm(10), f'{prefix}_top20': tm(20)}


def top_share(values, k):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return 0.0
    pos = np.maximum(values, 0.0); tot = float(pos.sum())
    if tot <= 1e-12:
        return 0.0
    return float(np.sort(pos)[::-1][:min(k, len(pos))].sum() / tot)


def top_entities(scores, labels, k):
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return [], [], {}
    idx = np.argsort(-scores)[:min(k, len(scores))]
    lab = [labels[i] for i in idx]; sc = [float(scores[i]) for i in idx]
    tot = float(np.maximum(sc, 0.0).sum())
    w = {l: 1.0 / max(len(lab), 1) for l in lab} if tot <= 1e-12 else {l: float(max(s, 0.0) / tot) for l, s in zip(lab, sc)}
    return lab, sc, w


def ecdf(reference, values):
    ref = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ref, np.asarray(values, dtype=float), side='right') / max(len(ref), 1)


def top_mean(values, k):
    values = np.asarray(values, dtype=float)
    return 0.0 if len(values) == 0 else float(np.sort(values)[::-1][:min(k, len(values))].mean())


def segments(frame):
    """Sort windows by (source_file, time, graph_id); new segment when gap > 7.5 min (history is reset)."""
    f = frame.sort_values(['source_file', 'win_start', 'graph_id'], kind='mergesort').reset_index(drop=True)
    gap = f.groupby('source_file', sort=False)['win_start'].diff().dt.total_seconds().div(60.0)
    new = gap.isna() | (gap <= 0.0) | (gap > MAX_GAP_MIN)
    f['segment'] = new.cumsum().astype(int)
    f['pos'] = f.groupby('segment').cumcount()
    return f


# ------------------------------------------------------------------ residual pass (D + node-local features of F)
RAW_NODE_NAMES = ['out_flow_count', 'in_flow_count', 'out_bytes', 'in_bytes', 'out_packets', 'in_packets', 'out_degree',
                  'in_degree', 'total_degree', 'unique_dst_ports', 'unique_src_ports', 'unique_protocols_out',
                  'unique_protocols_in', 'syn_out', 'syn_in', 'rst_out', 'rst_in', 'web_out_share', 'web_in_share',
                  'new_edge_out_share', 'new_edge_in_share', 'bytes_per_pkt_out', 'bytes_per_pkt_in', 'is_private_ip']


class NodeTable:
    """Column store for all nodes of the timeline windows (nodes of one window are contiguous)."""
    def __init__(self, cols, graph_ids, offsets, ip_codes, ip_names):
        self.c, self.graph_ids, self.off, self.ip, self.ip_names = cols, np.asarray(graph_ids), np.asarray(offsets), ip_codes, ip_names
        self.n = int(self.off[-1])
        self.gid = np.repeat(self.graph_ids, np.diff(self.off))

    def __getitem__(self, k):
        return self.c[k]

    def __setitem__(self, k, v):
        v = np.asarray(v)
        self.c[k] = np.full(self.n, v, dtype=np.float64) if v.ndim == 0 else v

    def slices(self):
        for g, a, b in zip(self.graph_ids, self.off[:-1], self.off[1:]):
            yield int(g), int(a), int(b)


def residual_pass(model, stats, graphs, node_ips, ids, fit_ids, schema, device):
    """One AE pass over the timeline windows -> window residual table, top entities per window, node table, profiles."""
    import torch
    node_index = {n: i for i, n in enumerate(schema['node_cols'])}
    edge_index_by_name = {n: i for i, n in enumerate(schema['edge_cols'])}

    @torch.no_grad()
    def forward(i):
        b = normalize(graphs[i], stats, device)
        xh, eh, gh = model(*b)
        return b, xh, eh, gh

    # 1) residual samples on FIT windows -> robust profiles
    ns, es = [], []
    for gid in fit_ids:
        b, xh, eh, _ = forward(gid)
        nl = torch.log1p(torch.abs(xh - b[0])).float().cpu().numpy()
        el = torch.log1p(torch.abs(eh - b[2])).float().cpu().numpy()
        ns.append(sample_rows(nl, NODE_SAMPLE_PER_FIT_GRAPH, 42 + int(gid) * 17 + 1))
        es.append(sample_rows(el, EDGE_SAMPLE_PER_FIT_GRAPH, 42 + int(gid) * 17 + 2))
    ns, es = np.concatenate(ns).astype(np.float32), np.concatenate(es).astype(np.float32)
    nstat, estat = fit_profile_stats(ns), fit_profile_stats(es)
    node_fit_entity = profile_components(ns, nstat)['score']; edge_fit_entity = profile_components(es, estat)['score']
    edge_q99 = float(np.quantile(edge_fit_entity, 0.99))
    log(f'profiles: node samples {len(ns)}, edge samples {len(es)}, edge q99 {edge_q99:.4f}')

    col = lambda m, idx, name: (np.asarray(m[:, idx[name]], dtype=np.float32) if name in idx else np.zeros(m.shape[0], np.float32))

    def isum(v, i, n):
        o = np.zeros(n)
        if len(v):
            np.add.at(o, i, v)
        return o

    def imax(v, i, n):
        o = np.zeros(n)
        if len(v):
            np.maximum.at(o, i, v)
        return o

    def sdiv(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-12)

    rows, ents, parts, sizes, ipc = [], [], [], [], []
    codes = {}
    for k, gid in enumerate(ids):
        b, xh, eh, gh = forward(gid)
        nabs, eabs = torch.abs(xh - b[0]), torch.abs(eh - b[2])
        node_mse = (nabs ** 2).mean(1).float().cpu().numpy(); edge_mse = (eabs ** 2).mean(1).float().cpu().numpy()
        graph_mse = ((gh - b[3]) ** 2)[0].float().cpu().numpy()
        gae_loss = float((nabs ** 2).mean() + (eabs ** 2).mean() + ((gh - b[3]) ** 2).mean())   # = AE training loss
        nl = torch.log1p(nabs).float().cpu().numpy(); el = torch.log1p(eabs).float().cpu().numpy()
        del b, xh, eh, gh, nabs, eabs
        nc, ec = profile_components(nl, nstat), profile_components(el, estat)
        nprof, eprof = nc['score'], ec['score']
        ips = [str(v) for v in node_ips[gid]]
        ei = graphs[gid]['edge_index'].numpy(); src, dst = ei[0], ei[1]
        rx, re_ = graphs[gid]['x'].numpy(), graphs[gid]['edge_raw'].numpy()
        n, m = rx.shape[0], re_.shape[0]
        assert len(ips) == n, f'node_ips size mismatch in window {gid}'
        pairs = [(ips[s], ips[t]) for s, t in zip(src, dst)]
        tnl, _, tnw = top_entities(nprof, ips, TOP_NODE_K)
        tel, _, tew = top_entities(eprof, pairs, TOP_EDGE_K)
        row = dict(graph_id=int(gid), gae_loss=gae_loss)
        row.update(aggregate_vector(node_mse, 'node')); row.update(aggregate_vector(edge_mse, 'edge'))
        row.update(aggregate_vector(graph_mse, 'graph'))
        row.update(aggregate_vector(nprof, 'node_res')); row.update(aggregate_vector(eprof, 'edge_res'))
        for kk in (1, 5, 10):
            row[f'node_res_top{kk}_share'] = top_share(nprof, kk); row[f'edge_res_top{kk}_share'] = top_share(eprof, kk)
        rows.append(row)
        ents.append(dict(graph_id=int(gid), top_node_labels=tnl, top_node_weights=tnw, top_edge_labels=tel, top_edge_weights=tew))

        # node-local features; float32 where the source is float32 (exactly representable in float64)
        oc, ic = isum(np.ones(m), src, n), isum(np.ones(m), dst, n)
        osum, isu = isum(eprof, src, n), isum(eprof, dst, n)
        omax, imx = imax(eprof, src, n), imax(eprof, dst, n)
        inc = oc + ic
        tail = (eprof > edge_q99).astype(np.float64)
        ec_ = {nm: col(re_, edge_index_by_name, nm) for nm in ('cnt', 'bytes_sum', 'pkts_sum', 'syn_sum', 'rst_sum', 'new_edge_flag', 'web_share', 'ssh_share', 'dns_share')}
        cnt1 = np.maximum(ec_['cnt'], 1.0)
        of, if_ = isum(ec_['cnt'], src, n), isum(ec_['cnt'], dst, n)
        ob, ib = isum(ec_['bytes_sum'], src, n), isum(ec_['bytes_sum'], dst, n)
        op, ip_ = isum(ec_['pkts_sum'], src, n), isum(ec_['pkts_sum'], dst, n)
        tot_f = float(np.maximum(ec_['cnt'], 0.0).sum()); tot_b = float(np.maximum(ec_['bytes_sum'], 0.0).sum())
        tot_p = float(np.maximum(ec_['pkts_sum'], 0.0).sum())
        nd = dict(
            node_profile_score=nprof, node_res_top3_mean=nc['top3_mean'], node_res_max=nc['max'], node_res_mean=nc['mean'],
            node_res_l2=nc['l2'], node_res_rare_fraction=nc['rare_fraction'], node_mse=node_mse,
            out_edge_residual_mean=sdiv(osum, oc), in_edge_residual_mean=sdiv(isu, ic), out_edge_residual_max=omax,
            in_edge_residual_max=imx, incident_edge_residual_mean=sdiv(osum + isu, inc), incident_edge_residual_max=np.maximum(omax, imx),
            incident_edge_tail_share=sdiv(isum(tail, src, n) + isum(tail, dst, n), inc),
            local_out_edge_count=oc, local_in_edge_count=ic, local_incident_edge_count=inc,
            local_out_flows=of, local_in_flows=if_, local_incident_flows=of + if_, local_out_bytes=ob, local_in_bytes=ib,
            local_incident_bytes=ob + ib, local_out_packets=op, local_in_packets=ip_, local_incident_packets=op + ip_,
            local_out_syn=isum(ec_['syn_sum'], src, n), local_in_syn=isum(ec_['syn_sum'], dst, n),
            local_out_rst=isum(ec_['rst_sum'], src, n), local_in_rst=isum(ec_['rst_sum'], dst, n),
            incident_flow_share=(of + if_) / max(tot_f, 1e-12), incident_bytes_share=(ob + ib) / max(tot_b, 1e-12),
            incident_packet_share=(op + ip_) / max(tot_p, 1e-12), degree_share=inc / max(float(m), 1.0),
            direction_balance=sdiv(np.abs(of - if_), of + if_),
            new_edge_share_local=sdiv(isum(ec_['new_edge_flag'], src, n) + isum(ec_['new_edge_flag'], dst, n), inc),
            web_share_local=sdiv(isum(ec_['web_share'] * cnt1, src, n) + isum(ec_['web_share'] * cnt1, dst, n), of + if_),
            ssh_share_local=sdiv(isum(ec_['ssh_share'] * cnt1, src, n) + isum(ec_['ssh_share'] * cnt1, dst, n), of + if_),
            dns_share_local=sdiv(isum(ec_['dns_share'] * cnt1, src, n) + isum(ec_['dns_share'] * cnt1, dst, n), of + if_),
            graph_n_nodes=np.full(n, float(n)), graph_n_edges=np.full(n, float(m)), graph_n_flows=np.full(n, tot_f))
        for nm in RAW_NODE_NAMES:
            nd['raw_' + nm] = col(rx, node_index, nm)
        parts.append(nd); sizes.append(n)
        ipc.append(np.fromiter((codes.setdefault(v, len(codes)) for v in ips), dtype=np.int64, count=n))
        if k == 0 or (k + 1) % 200 == 0 or k + 1 == len(ids):
            log(f'residual pass {k + 1}/{len(ids)}')
    cols = {}
    for c in list(parts[0].keys()):
        cols[c] = np.concatenate([p.pop(c) for p in parts])
    del parts
    names = np.empty(len(codes), dtype=object)
    for v, i in codes.items():
        names[i] = v
    N = NodeTable(cols, ids, np.concatenate([[0], np.cumsum(sizes)]), np.concatenate(ipc), names)
    prof = dict(node_median=nstat['median'], node_scale=nstat['scale'], edge_median=estat['median'], edge_scale=estat['scale'],
                node_fit_entity_scores=node_fit_entity, edge_fit_entity_scores=edge_fit_entity, edge_q99=edge_q99)
    return pd.DataFrame(rows), ents, N, prof


def desc_order(values, start):
    """Indices (global) of values sorted descending exactly as pandas DataFrame.sort_values(ascending=False)."""
    return pd.Series(values, index=np.arange(start, start + len(values))).sort_values(ascending=False).index.to_numpy()


# ------------------------------------------------------------------ temporal context
def entity_persistence(tl, ents, use_history):
    """Top-entity persistence over the previous 3 windows of the same segment (causal)."""
    by = {e['graph_id']: e for e in ents}
    cols = ['res_ip_jaccard_prev1', 'res_ip_fraction_seen3', 'res_ip_weighted_seen3', 'res_ip_max_repeat3', 'res_ip_top1_seen3',
            'res_ip_novel_fraction3', 'res_edge_jaccard_prev1', 'res_edge_fraction_seen3', 'res_edge_weighted_seen3',
            'res_edge_max_repeat3', 'res_edge_top1_seen3', 'res_edge_novel_fraction3']
    out = {c: np.zeros(len(tl)) for c in cols}
    if not use_history:
        out['res_ip_novel_fraction3'][:] = 1.0; out['res_edge_novel_fraction3'][:] = 1.0
        return pd.DataFrame(out, index=tl.index)
    jac = lambda a, b: 0.0 if not (set(a) | set(b)) else len(set(a) & set(b)) / len(set(a) | set(b))
    for _, g in tl.groupby('segment', sort=False):
        hist = deque(maxlen=HISTORY_WINDOWS)
        for ri, gid in zip(g.index, g.graph_id):
            cur = by[int(gid)]
            for kind, lab, wk in (('ip', 'top_node_labels', 'top_node_weights'), ('edge', 'top_edge_labels', 'top_edge_weights')):
                c = list(cur[lab]); cs = set(c)
                prev_sets = [set(h[lab]) for h in hist]; union = set().union(*prev_sets) if prev_sets else set()
                cnt = Counter()
                for s in prev_sets:
                    cnt.update(s)
                fs = 0.0 if not c else sum(e in union for e in c) / len(c)
                out[f'res_{kind}_jaccard_prev1'][ri] = jac(cs, hist[-1][lab]) if hist else 0.0
                out[f'res_{kind}_fraction_seen3'][ri] = fs
                out[f'res_{kind}_weighted_seen3'][ri] = 0.0 if not cur[wk] else sum(w for e, w in cur[wk].items() if e in union)
                out[f'res_{kind}_max_repeat3'][ri] = max([cnt.get(e, 0) for e in c] or [0]) / HISTORY_WINDOWS
                out[f'res_{kind}_top1_seen3'][ri] = float(c[0] in union) if c else 0.0
                out[f'res_{kind}_novel_fraction3'][ri] = 1.0 - fs
            hist.append(cur)
    return pd.DataFrame(out, index=tl.index)


def ip_history(N, tl, use_history):
    """GRPD per-IP context from the previous <=3 windows of the same segment (vectorised; causal)."""
    cols = ['ip_seen_prev1', 'ip_seen_count3', 'ip_score_prev1', 'ip_score_max3', 'ip_score_mean3', 'ip_top10_count3',
            'ip_score_jump_prev1', 'ip_score_jump_max3']
    cur = N['node_profile_score'].astype(np.float64)
    if not use_history:
        for c in cols:
            N[c] = np.zeros(N.n)
        N['ip_score_jump_prev1'] = cur.copy(); N['ip_score_jump_max3'] = cur.copy()
        return
    pos = tl.set_index('graph_id')
    seg = pos.loc[N.gid, 'segment'].to_numpy(np.int64); ps = pos.loc[N.gid, 'pos'].to_numpy(np.int64)
    top10 = np.zeros(N.n)
    for g, a, b in N.slices():
        top10[desc_order(N['node_profile_score'][a:b], a)[:TOP_IPS]] = 1.0
    base = pd.DataFrame(dict(segment=seg, pos=ps, ip=N.ip, s=cur, t=top10))
    key = base[['segment', 'pos', 'ip']]
    P, S, T = [], [], []
    for lag in (1, 2, 3):
        prev = base.assign(pos=base['pos'] + lag)
        mrg = key.merge(prev, on=['segment', 'pos', 'ip'], how='left', validate='one_to_one')
        p = mrg['s'].notna().to_numpy()
        P.append(p); S.append(np.where(p, mrg['s'].to_numpy(dtype=float), 0.0)); T.append(np.where(p, mrg['t'].to_numpy(dtype=float), 0.0))
        del mrg, prev
    P, S, T = np.vstack(P).T, np.vstack(S).T, np.vstack(T).T
    cnt = P.sum(1)
    smax = np.where(cnt > 0, np.max(np.where(P, S, -np.inf), axis=1), 0.0)
    smean = np.where(cnt > 0, (S * P).sum(1) / np.maximum(cnt, 1), 0.0)
    s1 = S[:, 0]
    N['ip_seen_prev1'] = P[:, 0].astype(float); N['ip_seen_count3'] = cnt.astype(float)
    N['ip_score_prev1'] = s1; N['ip_score_max3'] = smax; N['ip_score_mean3'] = smean
    N['ip_top10_count3'] = T.sum(1)
    N['ip_score_jump_prev1'] = cur - s1; N['ip_score_jump_max3'] = cur - smax


# ------------------------------------------------------------------ scoring
LOCAL_IF_FEATURES = ['node_profile_score', 'node_res_top3_mean', 'node_res_max', 'node_res_mean', 'node_res_l2', 'node_res_rare_fraction', 'node_mse',
    'out_edge_residual_mean', 'in_edge_residual_mean', 'out_edge_residual_max', 'in_edge_residual_max', 'incident_edge_residual_mean',
    'incident_edge_residual_max', 'incident_edge_tail_share', 'local_out_edge_count', 'local_in_edge_count', 'local_incident_edge_count',
    'local_out_flows', 'local_in_flows', 'local_incident_flows', 'local_out_bytes', 'local_in_bytes', 'local_incident_bytes',
    'local_out_packets', 'local_in_packets', 'local_incident_packets', 'local_out_syn', 'local_in_syn', 'local_out_rst', 'local_in_rst',
    'incident_flow_share', 'incident_bytes_share', 'incident_packet_share', 'degree_share', 'direction_balance', 'new_edge_share_local',
    'web_share_local', 'ssh_share_local', 'dns_share_local', 'ip_seen_prev1', 'ip_seen_count3', 'ip_score_prev1', 'ip_score_max3',
    'ip_score_mean3', 'ip_top10_count3', 'ip_score_jump_prev1', 'ip_score_jump_max3', 'raw_out_flow_count', 'raw_in_flow_count',
    'raw_out_degree', 'raw_in_degree', 'raw_unique_dst_ports', 'raw_unique_src_ports', 'raw_unique_protocols_out',
    'raw_unique_protocols_in', 'raw_syn_out', 'raw_syn_in', 'raw_rst_out', 'raw_rst_in', 'raw_web_out_share', 'raw_web_in_share',
    'raw_new_edge_out_share', 'raw_new_edge_in_share', 'raw_bytes_per_pkt_out', 'raw_bytes_per_pkt_in', 'raw_is_private_ip',
    'graph_n_nodes', 'graph_n_edges', 'graph_n_flows']
LOG_FEATURES = {'node_mse', 'local_out_edge_count', 'local_in_edge_count', 'local_incident_edge_count', 'local_out_flows', 'local_in_flows',
    'local_incident_flows', 'local_out_bytes', 'local_in_bytes', 'local_incident_bytes', 'local_out_packets', 'local_in_packets',
    'local_incident_packets', 'local_out_syn', 'local_in_syn', 'local_out_rst', 'local_in_rst', 'raw_out_flow_count', 'raw_in_flow_count',
    'raw_out_degree', 'raw_in_degree', 'raw_unique_dst_ports', 'raw_unique_src_ports', 'raw_unique_protocols_out', 'raw_unique_protocols_in',
    'raw_syn_out', 'raw_syn_in', 'raw_rst_out', 'raw_rst_in', 'raw_bytes_per_pkt_out', 'raw_bytes_per_pkt_in', 'graph_n_nodes',
    'graph_n_edges', 'graph_n_flows'}
IP_HISTORY_FEATURES = [c for c in LOCAL_IF_FEATURES if c.startswith('ip_')]


def window_channels(W, tl, ents, val_ids, use_history):
    """Window channels and the residual entity gate."""
    W = W.set_index('graph_id').loc[tl.graph_id].reset_index()
    val = W.graph_id.isin(val_ids).to_numpy()
    base = ['node_p95', 'node_p99', 'node_top10', 'node_top20', 'node_max']
    nres = ['node_res_p90', 'node_res_p95', 'node_res_p99', 'node_res_top5', 'node_res_top10', 'node_res_top20', 'node_res_max']
    eres = [c.replace('node', 'edge') for c in nres]
    conc = ['node_res_top1_share', 'node_res_top5_share', 'node_res_top10_share', 'edge_res_top1_share', 'edge_res_top5_share', 'edge_res_top10_share']
    E = {c: ecdf(W.loc[val, c], W[c]) for c in base + nres + eres + conc}
    mx = lambda cs: np.max(np.vstack([E[c] for c in cs]), axis=0)
    W['S_node_local_hi'], W['S_node_residual_profile'] = mx(base), mx(nres)
    W['S_edge_residual_profile'], W['S_residual_concentration'] = mx(eres), mx(conc)
    P = entity_persistence(tl, ents, use_history)
    P.index = W.index
    pe = {c: ecdf(P.loc[val, c], P[c]) for c in P.columns}
    ipp = np.max(np.vstack([pe[f'res_ip_{s}'] for s in ('jaccard_prev1', 'fraction_seen3', 'weighted_seen3', 'max_repeat3', 'top1_seen3')]), axis=0)
    edp = np.max(np.vstack([pe[f'res_edge_{s}'] for s in ('jaccard_prev1', 'fraction_seen3', 'weighted_seen3', 'max_repeat3', 'top1_seen3')]), axis=0)
    W['E_res_entity_persistence'] = (CFG['persist_w'][0] * ipp + CFG['persist_w'][1] * edp) if use_history else 0.0
    w = CFG['res_w']
    W['S_residual_fusion'] = (w[0] * W['S_node_residual_profile'] + w[1] * W['S_edge_residual_profile']
                              + w[2] * W['S_node_local_hi'] + w[3] * W['S_residual_concentration'])
    W['S_residual_entity_gate'] = W['S_residual_fusion'] * ((1 - CFG['gate_g']) + CFG['gate_g'] * W['E_res_entity_persistence'])
    return W


def node_scores(N, fit_ids, use_history, seed, threads):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler
    fitm = np.isin(N.gid, fit_ids)
    fit_set = set(int(g) for g in fit_ids)
    sel = []
    for g, a, b in sorted(N.slices()):
        if g not in fit_set:
            continue
        if b - a <= MAX_FIT_NODES_PER_GRAPH:
            sel.extend(range(a, b)); continue
        order = desc_order(N['node_profile_score'][a:b], a)
        tail, rest = order[:CFG['tail_nodes']], order[CFG['tail_nodes']:]
        rnd = np.random.default_rng(42 + int(g) * 31).choice(rest, size=MAX_FIT_NODES_PER_GRAPH - len(tail), replace=False)
        sel.extend(tail.tolist()); sel.extend(rnd.tolist())
    sel = np.asarray(sel, dtype=int)
    feats = [c for c in LOCAL_IF_FEATURES if use_history or c not in IP_HISTORY_FEATURES]

    def matrix(rows):
        M = np.empty((len(rows), len(feats)), dtype=np.float64)
        for j, c in enumerate(feats):
            v = N[c][rows].astype(np.float64)
            if c in LOG_FEATURES:
                v = np.log1p(np.clip(v, 0.0, None))
            M[:, j] = v
        M[~np.isfinite(M)] = 0.0
        return M
    scaler = RobustScaler(quantile_range=CFG['node_scaler_q'])
    Xs = scaler.fit_transform(matrix(sel))
    ms = CFG['if_max_samples'] if isinstance(CFG['if_max_samples'], str) else min(CFG['if_max_samples'], len(Xs))
    iso = IsolationForest(n_estimators=CFG['if_trees'], max_samples=ms,
                          max_features=CFG['if_max_features'], contamination='auto', random_state=seed, n_jobs=threads)
    iso.fit(Xs)
    raw = np.empty(N.n)
    step = 500_000
    for a in range(0, N.n, step):
        idx = np.arange(a, min(a + step, N.n))
        raw[idx] = -iso.decision_function(scaler.transform(matrix(idx)))
    N['local_if_raw'] = raw
    f64 = lambda c: N[c].astype(np.float64)
    N['S_local_if_node'] = ecdf(raw[fitm], raw)
    N['S_local_profile_node'] = ecdf(f64('node_profile_score')[fitm], f64('node_profile_score'))
    N['S_local_edge_node'] = ecdf(f64('incident_edge_residual_max')[fitm], f64('incident_edge_residual_max'))
    T = (0.25 * N['ip_seen_prev1'] + 0.25 * (N['ip_seen_count3'] / HISTORY_WINDOWS) + 0.25 * (N['ip_top10_count3'] / HISTORY_WINDOWS)
         + 0.25 * ecdf(N['ip_score_max3'][fitm], N['ip_score_max3']))
    N['T_ip_persistence'] = np.clip(T, 0.0, 1.0) if use_history else np.zeros(N.n)
    w = CFG['node_w']
    N['S_local_fusion_node'] = w[0] * N['S_local_profile_node'] + w[1] * N['S_local_if_node'] + w[2] * N['S_local_edge_node'] + w[3] * N['T_ip_persistence']
    N['S_local_temporal_node'] = N['S_local_fusion_node'] * ((1 - CFG['temporal_g']) + CFG['temporal_g'] * N['T_ip_persistence'])
    N['S_local_rescue_node'] = np.maximum.reduce([CFG['rescue_w'][0] * N['S_local_profile_node'], CFG['rescue_w'][1] * N['S_local_if_node'], N['S_local_temporal_node']])
    q99 = float(np.quantile(N['S_local_profile_node'][fitm], 0.99))
    return q99, len(sel), len(feats)


def window_scores(N, W, val_ids, q99):
    cols = ['S_local_profile_node', 'S_local_if_node', 'S_local_fusion_node', 'S_local_temporal_node', 'S_local_rescue_node']
    rows, top = [], []
    for g, a, b in N.slices():
        r = dict(graph_id=g, local_node_count=b - a)
        for c in cols:
            v = N[c][a:b].astype(float); p = c.replace('_node', '')
            r[f'{p}_max'] = float(v.max()); r[f'{p}_p99'] = float(np.quantile(v, 0.99))
            r[f'{p}_top3'], r[f'{p}_top5'], r[f'{p}_top10'] = top_mean(v, 3), top_mean(v, 5), top_mean(v, 10)
        r['local_nodes_above_fit_q99'] = int((N['S_local_profile_node'][a:b] > q99).sum())
        r['local_nodes_above_fit_q99_share'] = float(r['local_nodes_above_fit_q99'] / max(b - a, 1))
        rows.append(r)
        o = desc_order(N['S_local_rescue_node'][a:b], a)[:TOP_IPS]
        top.append(pd.DataFrame(dict(graph_id=g, rank=np.arange(1, len(o) + 1), ip=N.ip_names[N.ip[o]],
                                     **{c: N[c][o] for c in ('S_local_rescue_node', 'S_local_profile_node', 'S_local_if_node',
                                                             'S_local_edge_node', 'T_ip_persistence')})))
    G = pd.DataFrame(rows).merge(W, on='graph_id', how='inner', validate='one_to_one')
    val = G.graph_id.isin(val_ids).to_numpy()
    comps = ['S_local_profile_max', 'S_local_profile_p99', 'S_local_profile_top3', 'S_local_profile_top5', 'S_local_if_max', 'S_local_if_p99',
             'S_local_if_top3', 'S_local_if_top5', 'S_local_temporal_max', 'S_local_temporal_p99', 'S_local_temporal_top3', 'S_local_temporal_top5',
             'S_local_rescue_max', 'S_local_rescue_p99', 'S_local_rescue_top3', 'S_local_rescue_top5', 'local_nodes_above_fit_q99',
             'local_nodes_above_fit_q99_share']
    E = {c: ecdf(G.loc[val, c], G[c]) for c in comps}
    mx = lambda cs: np.max(np.vstack([E[c] for c in cs]), axis=0)
    G['S_local_profile_graph'] = mx(['S_local_profile_max', 'S_local_profile_p99', 'S_local_profile_top3', 'S_local_profile_top5',
                                     'local_nodes_above_fit_q99', 'local_nodes_above_fit_q99_share'])
    G['S_local_if_graph'] = mx(['S_local_if_max', 'S_local_if_p99', 'S_local_if_top3', 'S_local_if_top5'])
    G['S_local_temporal_graph'] = mx(['S_local_temporal_max', 'S_local_temporal_p99', 'S_local_temporal_top3', 'S_local_temporal_top5'])
    G['S_local_rescue_graph'] = mx(['S_local_rescue_max', 'S_local_rescue_p99', 'S_local_rescue_top3', 'S_local_rescue_top5', 'local_nodes_above_fit_q99'])
    w = CFG['local_w']
    G['S_local_subgraph'] = (w[0] * G['S_local_profile_graph'] + w[1] * G['S_local_if_graph'] + w[2] * G['S_local_temporal_graph']
                             + w[3] * G['S_local_rescue_graph'])
    G['S_grpd'] = CFG['final_w'][0] * G['S_residual_entity_gate'] + CFG['final_w'][1] * G['S_local_subgraph']
    return G, pd.concat(top, ignore_index=True)


# ------------------------------------------------------------------ main
def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--split', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--encoder', choices=['sage', 'mlp'], default='sage')
    ap.add_argument('--graph-attr', choices=['full', 'base'], default='full'); ap.add_argument('--no-ip-history', action='store_true')
    ap.add_argument('--timeline', choices=['all', 'roles'], default='all', help='all: history over every built window (used in the paper)')
    ap.add_argument('--ae-from', default=None)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', '16'))); ap.add_argument('--device', default=None)
    ap.add_argument('--save-nodes', action='store_true', help='also save the full node table (large)')
    ap.add_argument('--max-epochs', type=int, default=AE['epochs'], help='only for smoke tests')
    ap.add_argument('--no-edge-attr', action='store_true', help='ablation: autoencoder without the 46 edge attributes')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    AE['epochs'] = a.max_epochs
    if device_is_cuda := torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # full fp32
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); LOG = out / 'log.txt'
    if (out / 'DONE').exists():
        log('already DONE, nothing to do'); return
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = dict(vars(a), device=device, torch=torch.__version__, started=time.strftime('%Y-%m-%d %H:%M:%S'),
               ae=AE, history=not a.no_ip_history, cfg=CFG)
    (out / 'config.json').write_text(json.dumps(cfg, indent=1))
    log(f'config {cfg}')

    global MAX_GAP_MIN
    try:
        import resource            # not available on Windows
    except ImportError:
        resource = None
    T0 = time.time(); timing = {}
    meta, graphs, node_ips, schema = load_data(a.data, a.graph_attr, a.no_edge_attr)
    win_min = ((pd.to_datetime(meta.win_end) - meta.win_start).median().total_seconds() / 60) if 'win_end' in meta else 5.0
    MAX_GAP_MIN = 1.5 * win_min       # history is reset after a gap of more than 1.5 windows (7.5 min for 5-min windows)
    timing['load_s'] = time.time() - T0; timing['window_min'] = win_min
    split = pd.read_csv(a.split)
    role = dict(zip(split.graph_id, split.role))
    ids_of = lambda r: np.array(sorted(g for g, x in role.items() if x == r), dtype=int)
    fit_ids, cal_ids, val_ids = ids_of('FIT'), ids_of('CAL'), ids_of('VAL')
    if len(cal_ids) == 0:          # LODO splits: early stopping on the VAL windows
        cal_ids = val_ids
    log(f'windows: all {len(meta)}, FIT {len(fit_ids)}, CAL {len(cal_ids)}, VAL {len(val_ids)}, '
        f'TEST {len(ids_of("TEST"))}, roles {split.role.value_counts().to_dict()}')

    # A. autoencoder
    ae_dir = Path(a.ae_from) / 'ae' if a.ae_from else out / 'ae'
    if not (ae_dir / 'model_best.pt').exists() or (not a.ae_from and not (ae_dir / 'DONE').exists()):
        assert not a.ae_from, f'--ae-from has no trained AE: {ae_dir}'
        log('training AE'); t = time.time()
        train_ae(graphs, fit_ids, cal_ids, ae_dir, a.encoder, a.seed, device)
        timing['ae_train_s'] = time.time() - t
        log(f'AE trained in {time.time() - t:.0f}s')
    model, stats = load_ae(ae_dir, device)

    # timeline
    if a.timeline == 'roles':
        tl_ids = sorted(g for g, x in role.items() if x in ('FIT', 'CAL', 'VAL', 'TEST'))
    else:
        tl_ids = meta.graph_id.tolist()
    tl = segments(meta.loc[meta.graph_id.isin(tl_ids), ['graph_id', 'source_file', 'win_start']])
    log(f'timeline {a.timeline}: {len(tl)} windows, {tl.segment.nunique()} segments')

    # D + node-local features
    t = time.time()
    W, ents, N, prof = residual_pass(model, stats, graphs, node_ips, sorted(tl_ids), fit_ids, schema, device)
    del graphs
    np.savez_compressed(out / 'profiles.npz', **{k: np.asarray(v) for k, v in prof.items()})
    timing['residual_pass_s'] = time.time() - t; timing['windows_scored'] = len(tl_ids); timing['nodes'] = int(N.n)
    log(f'residual pass {time.time() - t:.0f}s, nodes {N.n}')
    use_hist = not a.no_ip_history
    Wc = window_channels(W, tl, ents, val_ids, use_hist)
    ip_history(N, tl, use_hist)
    t = time.time()
    q99, n_sample, n_feat = node_scores(N, fit_ids, use_hist, a.seed, a.threads)
    timing['local_if_fit_and_score_s'] = time.time() - t
    log(f'local IF on {n_sample} FIT nodes x {n_feat} features, {time.time() - t:.0f}s')
    G, top = window_scores(N, Wc, val_ids, q99)
    G = G.merge(meta[['graph_id', 'source_file', 'win_start', 'n_nodes', 'n_edges', 'n_flows_raw']], on='graph_id', how='left')
    G.to_parquet(out / 'window_scores.parquet', index=False)
    top.to_parquet(out / 'node_top.parquet', index=False)
    if a.save_nodes:   # column by column (no pandas copy of the whole table)
        import pyarrow as pa, pyarrow.parquet as pq
        names = ['graph_id', 'ip'] + list(N.c.keys())
        arrays = [pa.array(N.gid), pa.array(N.ip_names[N.ip].astype(str))] + [pa.array(N.c[k]) for k in N.c]
        pq.write_table(pa.Table.from_arrays(arrays, names=names), out / 'node_scores.parquet')
    timing['total_s'] = time.time() - T0
    timing['scoring_s_per_window'] = (timing['residual_pass_s'] + timing['local_if_fit_and_score_s']) / max(len(tl_ids), 1)
    timing['peak_rss_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6 if resource else None
    timing['peak_gpu_gb'] = (torch.cuda.max_memory_allocated() / 1e9) if device == 'cuda' else 0.0
    timing['device'] = device; timing['gpu'] = torch.cuda.get_device_name(0) if device == 'cuda' else ''
    timing['threads'] = a.threads
    (out / 'timing.json').write_text(json.dumps(timing, indent=1))
    (out / 'DONE').write_text(json.dumps(dict(finished=time.strftime('%Y-%m-%d %H:%M:%S'), windows=len(G), nodes=N.n)))
    log('DONE')


if __name__ == '__main__':
    main()
