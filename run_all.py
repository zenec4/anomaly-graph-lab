"""Run all experiments of the paper (resumable job scheduler for one machine with one or more GPUs).

Usage:
  python run_all.py --csv-dir <dir with the corrected CSV files> --root <output dir>
                    [--datasets cic2018 cic2017] [--stages day lodo windows] [--gpus 1] [--jobs-per-gpu 2]
                    [--threads 8] [--flow-workers 16] [--lodo-flow-2018] [--smoke]

Stages (per dataset):
  day      5-min graphs, day-disjoint split and random within-day split:
           GRPD (5 seeds), GRPD with an MLP encoder (3 seeds), GRPD on the random split (3 seeds),
           Anomal-E (3 seeds), PCA edge-reconstruction control, DOMINANT and OCGNN (3 seeds each),
           window-vector and flow-level one-class baselines (both splits)          -> <root>/<ds>/eval/day
  lodo     leave-one-day-out folds on the 5-min graphs: GRPD, Anomal-E, DOMINANT, OCGNN (3 seeds each),
           PCA control, window-vector baselines; flow-level baselines for CIC-IDS2017
           (and for CSE-CIC-IDS2018 with --lodo-flow-2018)                         -> <root>/<ds>/eval/lodo
  windows  1, 2, 10 and 15-min graphs, day-disjoint split: GRPD (3 seeds) and window-vector baselines at every
           length; Anomal-E, PCA control, DOMINANT and OCGNN at 1 min              -> <root>/<ds>/eval/w<k>
The flow count per window is evaluated automatically in every evaluation.
A job is skipped when its DONE marker exists, so the same command resumes after an interruption.
--smoke: CIC-IDS2017 Monday and Thursday only, 2 epochs, one seed (checks the installation in a few minutes).
Logs: <root>/logs/<job>.log; progress: <root>/run_all.log
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from datasets import DATASETS

PY = sys.executable


def build_jobs(ds, a, root, w):
    """Graph construction for one window length: one job per day, then merge (+ LODO splits at 5 min)."""
    cfg = DATASETS[ds]
    g = root / ds / ('graphs' if w == 5 else f'graphs_w{w}')
    J, names = [], []
    for d in cfg['days']:
        stem = Path(d).stem
        J.append(dict(name=f'{ds}:w{w}:build:{stem}', deps=[], gpu=False, done=g / 'days' / f'{stem}.done',
                      cmd=[PY, HERE / 'build_day.py', Path(a.csv_dir) / d, g / 'days', '--window', str(w)]))
        names.append(J[-1]['name'])
    J.append(dict(name=f'{ds}:w{w}:merge', deps=names, gpu=False, done=g / 'data' / 'split_summary.json',
                  cmd=[PY, HERE / 'merge_build.py', ds, g / 'days', g / 'data', str(w)]))
    if w == 5:
        J.append(dict(name=f'{ds}:w5:lodo_splits', deps=[f'{ds}:w5:merge'], gpu=False, done=g / 'data' / 'lodo_summary.json',
                      cmd=[PY, HERE / 'make_lodo.py', g / 'data']))
    return J, g / 'data'


def detector_jobs(prefix, data, split, rdir, a, smoke, dep, seeds, which, tag=''):
    """Jobs for one split. which: subset of grpd, grpd_mlp, grpd_random, anomale, pcarec, node, bl_graph, bl_flow."""
    ep = ['--max-epochs', '2'] if smoke else []
    J = []

    def add(name, cmd, gpu=True, heavy=False):
        J.append(dict(name=f'{prefix}:{tag}{name}', deps=dep, gpu=gpu, heavy=heavy, done=rdir / f'{tag}{name}' / 'DONE', cmd=cmd))

    grpd = lambda out, s, sp, *extra: [PY, HERE / 'grpd.py', '--data', data, '--split', sp, '--out', out, '--seed', str(s),
                                       '--threads', str(a.threads), *extra] + ep
    for s in seeds.get('grpd', []):
        if 'grpd' in which:
            add(f'grpd_s{s}', grpd(rdir / f'{tag}grpd_s{s}', s, split))
    for s in seeds.get('aux', []):
        if 'grpd_mlp' in which:
            add(f'grpd_mlp_s{s}', grpd(rdir / f'{tag}grpd_mlp_s{s}', s, split, '--encoder', 'mlp'))
        if 'grpd_random' in which:
            add(f'grpd_random_s{s}', grpd(rdir / f'{tag}grpd_random_s{s}', s, data / 'split_random.csv'))
        if 'anomale' in which:
            add(f'anomale_s{s}', [PY, HERE / 'baselines_edge.py', data, split, rdir / f'{tag}anomale_s{s}', '--method', 'anomale',
                                  '--seed', str(s), '--threads', str(a.threads)] + ep)
        if 'node' in which:
            for m in ('dominant', 'ocgnn'):
                add(f'{m}_s{s}', [PY, HERE / 'baselines_node.py', data, split, rdir / f'{tag}{m}_s{s}', '--method', m,
                                  '--seed', str(s), '--threads', str(a.threads)] + ep)
    if 'pcarec' in which:
        add('pcarec', [PY, HERE / 'baselines_edge.py', data, split, rdir / f'{tag}pcarec', '--method', 'pcarec'], gpu=False)
    if 'bl_graph' in which:
        add('bl_graph', [PY, HERE / 'baselines_graph.py', data, split, rdir / f'{tag}bl_graph'], gpu=False)
        if 'grpd_random' in which:
            add('bl_graph_random', [PY, HERE / 'baselines_graph.py', data, data / 'split_random.csv', rdir / f'{tag}bl_graph_random'], gpu=False)
    if 'bl_flow' in which and not smoke:
        ds = prefix.split(':')[0]
        add('bl_flow', [PY, HERE / 'baselines_flow.py', ds, a.csv_dir, data, split, rdir / f'{tag}bl_flow',
                        '--workers', str(a.flow_workers)], gpu=False, heavy=True)
        if 'grpd_random' in which:
            add('bl_flow_random', [PY, HERE / 'baselines_flow.py', ds, a.csv_dir, data, data / 'split_random.csv',
                                   rdir / f'{tag}bl_flow_random', '--workers', str(a.flow_workers)], gpu=False, heavy=True)
    return J


def jobs_for(ds, a, root, smoke):
    cfg = DATASETS[ds]
    s5, s3 = ([42], [42]) if smoke else ([42, 43, 44, 45, 46], [42, 43, 44])
    boot = '200' if smoke else '1000'
    J = []
    b5, data = build_jobs(ds, a, root, 5)
    J += b5
    if 'day' in a.stages:
        rdir = root / ds / 'runs' / 'day'
        dj = detector_jobs(f'{ds}:day', data, data / 'split_day.csv', rdir, a, smoke, [f'{ds}:w5:merge'],
                           dict(grpd=s5, aux=s3), {'grpd', 'grpd_mlp', 'grpd_random', 'anomale', 'pcarec', 'node', 'bl_graph', 'bl_flow'})
        J += dj
        J.append(dict(name=f'{ds}:day:eval', deps=[j['name'] for j in dj], soft=True, gpu=False, done=root / ds / 'eval' / 'day' / 'metrics.csv',
                      cmd=[PY, HERE / 'evaluate.py', data, rdir, root / ds / 'eval' / 'day', '--boot', boot, '--main', 'grpd_s42']))
    if 'lodo' in a.stages:
        rdir = root / ds / 'runs' / 'lodo'; lj = []
        which = {'grpd', 'anomale', 'pcarec', 'node', 'bl_graph'}
        if ds != 'cic2018' or a.lodo_flow_2018:
            which.add('bl_flow')
        for d in cfg['days']:
            if d in cfg.get('benign_only_days', []):
                continue
            stem = Path(d).stem
            lj += detector_jobs(f'{ds}:lodo', data, data / f'split_lodo_{stem}.csv', rdir, a, smoke, [f'{ds}:w5:lodo_splits'],
                                dict(grpd=s3, aux=s3), which, tag=f'{stem}__')
        J += lj
        J.append(dict(name=f'{ds}:lodo:summary', deps=[j['name'] for j in lj], soft=True, gpu=False, done=root / ds / 'eval' / 'lodo' / 'pooled.csv',
                      cmd=[PY, HERE / 'lodo_summary.py', data, rdir, root / ds / 'eval' / 'lodo']))
    if 'windows' in a.stages:
        for w in ([1] if smoke else [1, 2, 10, 15]):
            bw, dw = build_jobs(ds, a, root, w)
            J += bw
            rdir = root / ds / 'runs' / f'w{w}'
            which = {'grpd', 'bl_graph'} | ({'anomale', 'pcarec', 'node'} if w == 1 else set())
            wj = detector_jobs(f'{ds}:w{w}', dw, dw / 'split_day.csv', rdir, a, smoke, [f'{ds}:w{w}:merge'], dict(grpd=s3, aux=s3), which)
            J += wj
            J.append(dict(name=f'{ds}:w{w}:eval', deps=[j['name'] for j in wj], soft=True, gpu=False,
                          done=root / ds / 'eval' / f'w{w}' / 'metrics.csv',
                          cmd=[PY, HERE / 'evaluate.py', dw, rdir, root / ds / 'eval' / f'w{w}', '--boot', '0', '--main', 'grpd_s42']))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-dir', required=True); ap.add_argument('--root', required=True)
    ap.add_argument('--datasets', nargs='+', default=['cic2018', 'cic2017'])
    ap.add_argument('--stages', nargs='+', choices=['day', 'lodo', 'windows'], default=['day', 'lodo', 'windows'])
    ap.add_argument('--gpus', type=int, default=1); ap.add_argument('--jobs-per-gpu', type=int, default=2)
    ap.add_argument('--threads', type=int, default=8); ap.add_argument('--flow-workers', type=int, default=16)
    ap.add_argument('--max-cpu-jobs', type=int, default=8); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--lodo-flow-2018', action='store_true', help='also run the flow-level baselines in the CSE-CIC-IDS2018 LODO folds (slow)')
    a = ap.parse_args()
    root = Path(a.root); (root / 'logs').mkdir(parents=True, exist_ok=True)
    datasets = ['smoke2017'] if a.smoke else a.datasets
    J, seen = [], set()
    for ds in datasets:
        for j in jobs_for(ds, a, root, a.smoke):
            if j['name'] not in seen:
                seen.add(j['name']); J.append(j)
    (root / 'run_all_config.json').write_text(json.dumps(dict(vars(a), jobs=[j['name'] for j in J]), indent=1))
    plog = open(root / 'run_all.log', 'a', buffering=1)

    def say(m):
        print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
        plog.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {m}\n')

    done = {j['name'] for j in J if Path(j['done']).exists()}
    failed, running = set(), {}
    gpu_load = {i: 0 for i in range(max(a.gpus, 1))}
    say(f'{len(J)} jobs, {len(done)} already done')
    while True:
        for name, (p, j, gpu, fh) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            fh.close(); del running[name]
            if gpu is not None:
                gpu_load[gpu] -= 1
            if rc == 0 and Path(j['done']).exists():
                done.add(name); say(f'DONE   {name}')
            else:
                failed.add(name); say(f'FAILED {name} (rc={rc}), see logs/{name.replace(":", "_")}.log')
        pending = [j for j in J if j['name'] not in done | failed | set(running)]
        # a failed dependency skips the job, except for 'soft' jobs (evaluations run on whatever finished)
        blocked = [j for j in pending if not j.get('soft') and any(d in failed for d in j['deps'])]
        for j in blocked:
            failed.add(j['name']); say(f'SKIP   {j["name"]} (dependency failed)')
        pending = [j for j in pending if j not in blocked]
        if not pending and not running:
            break
        n_cpu = sum(1 for _, j, g, _ in running.values() if g is None)
        heavy_running = any(j.get('heavy') for _, j, _, _ in running.values())
        for j in pending:
            if not all(d in done or (j.get('soft') and d in failed) for d in j['deps']):
                continue
            env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), PYTHONUNBUFFERED='1')
            gpu = None
            if j['gpu'] and a.gpus > 0:
                gpu = min(gpu_load, key=gpu_load.get)
                if gpu_load[gpu] >= a.jobs_per_gpu:
                    continue
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            else:
                if n_cpu >= a.max_cpu_jobs or (j.get('heavy') and heavy_running):
                    continue
                if not j['gpu']:
                    env['CUDA_VISIBLE_DEVICES'] = ''
                if j.get('heavy'):
                    env['OMP_NUM_THREADS'] = '1'; heavy_running = True
                n_cpu += 1
            fh = open(root / 'logs' / (j['name'].replace(':', '_') + '.log'), 'a')
            p = subprocess.Popen([str(c) for c in j['cmd']], stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
            running[j['name']] = (p, j, gpu, fh)
            if gpu is not None:
                gpu_load[gpu] += 1
            say(f'START  {j["name"]}' + (f' on GPU {gpu}' if gpu is not None else ''))
        time.sleep(5)
    say(f'FINISHED: {len(done)} done, {len(failed)} failed' + (f': {sorted(failed)}' if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
