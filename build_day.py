"""Build the 5-minute IP-interaction graphs of ONE capture day (one CSV of the Improved CIC datasets).

Usage: python build_day.py <day.csv> <out_dir> [--window 5]
Writes <out_dir>/<day>.pt (list of dicts x, edge_index, edge_attr, graph_attr), <day>_meta.parquet (window
statistics + label columns), <day>_ips.pkl (node index -> IP per window), <day>.done.
Feature code: graph_builder.py. Windows with < 20 nodes or < 20 edges are skipped.
"""
import argparse, gc, json, pickle, sys, time
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_builder as gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv'); ap.add_argument('out')
    ap.add_argument('--chunksize', type=int, default=500_000)
    ap.add_argument('--window', type=int, default=5, help='window length in minutes (default 5)')
    a = ap.parse_args()
    gb.WINDOW = f'{a.window}min'
    csv, out = Path(a.csv), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    day = csv.stem
    if (out / f'{day}.done').exists():
        print('already built', day); return
    t0 = time.time()
    win2parts, rows = defaultdict(list), 0
    for i, chunk in enumerate(pd.read_csv(csv, chunksize=a.chunksize, low_memory=False), 1):
        c = gb.canonicalize_chunk(chunk, csv.name)
        if len(c) == 0:
            continue
        c['win_start'] = c['timestamp'].dt.floor(gb.WINDOW)
        rows += len(c)
        for ws, part in c.groupby('win_start', sort=False):
            win2parts[pd.Timestamp(ws)].append(part.drop(columns=['win_start']).reset_index(drop=True))
        if i % 10 == 0:
            print(f'{day}: chunks {i}, rows {rows:,}, windows {len(win2parts)}', flush=True)
        del chunk, c
    print(f'{day}: read {rows:,} flows into {len(win2parts)} windows in {time.time() - t0:.0f}s', flush=True)

    history_edges, history_services, graph_history = set(), set(), deque(maxlen=gb.GRAPH_HISTORY_LEN)
    graphs, metas, ips = [], [], []
    for k, ws in enumerate(sorted(win2parts)):
        df = pd.concat(win2parts.pop(ws), ignore_index=True)
        data, meta, mapping = gb.make_graph_from_window(df_raw=df, source_file=csv.name, win_start=ws, graph_global_id=len(graphs),
                                                        history_edges=history_edges, history_services=history_services,
                                                        graph_history=graph_history)
        if data is None:
            continue
        graphs.append(dict(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, graph_attr=data.graph_attr))
        metas.append(meta)
        ips.append([mapping['node_idx_to_ip'][j] for j in range(data.num_nodes)])
        if len(graphs) % 50 == 0:
            print(f'{day}: {len(graphs)} graphs ({k + 1} windows seen)', flush=True)
        del df
        gc.collect()
    torch.save(graphs, out / f'{day}.pt')
    pd.DataFrame(metas).to_parquet(out / f'{day}_meta.parquet', index=False)
    with open(out / f'{day}_ips.pkl', 'wb') as f:
        pickle.dump(ips, f, protocol=pickle.HIGHEST_PROTOCOL)
    schema = dict(version=gb.VERSION, node_cols=gb.NODE_COLS, edge_cols=gb.EDGE_COLS, graph_base_cols=gb.GRAPH_BASE_COLS,
                  graph_attr_layout=['current', 'delta_prev', 'z_history', 'relative_to_history_mean'],
                  window=gb.WINDOW, min_nodes=gb.MIN_NODES, min_edges=gb.MIN_EDGES, graph_history_len=gb.GRAPH_HISTORY_LEN)
    (out / 'feature_schema.json').write_text(json.dumps(schema, indent=1))
    (out / f'{day}.done').write_text(json.dumps(dict(graphs=len(graphs), flows=rows, seconds=round(time.time() - t0))))
    print(f'{day}: DONE {len(graphs)} graphs in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
