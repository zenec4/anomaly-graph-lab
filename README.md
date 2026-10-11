# Held-out-day evaluation of benign-only graph intrusion detectors

Code, splits and evaluation scripts for the paper
*Held-Out-Day Evaluation of Benign-Only Graph Intrusion Detectors on the Corrected CIC Datasets*.

The code builds one IP interaction graph per time window from the corrected CIC-IDS2017 and CSE-CIC-IDS2018 flow
files. It then trains benign-only detectors on these graphs and evaluates them under day-disjoint,
leave-one-day-out (LODO) and random within-day splits.

## Contents

| File | Purpose |
|---|---|
| `datasets.py` | Day lists and day-disjoint roles of both datasets |
| `graph_builder.py`, `build_day.py` | Window graphs of one capture day: 30 node, 46 edge and 188 window-attribute features |
| `merge_build.py` | Merges the days and writes window labels and splits (day-disjoint and random within-day) |
| `make_lodo.py` | Leave-one-day-out splits |
| `grpd.py` | GRPD: graph autoencoder, per-feature residual profiles and node-local isolation forest |
| `baselines_edge.py` | Anomal-E (re-implementation) and the PCA edge-reconstruction control |
| `baselines_node.py` | DOMINANT and OCGNN (re-implementations for inductive training on window graphs) |
| `baselines_graph.py` | 14 one-class models on window feature vectors |
| `baselines_flow.py` | 20 flow-level one-class configurations aggregated per window |
| `evaluate.py` | Metrics for one split: AUC, macro-F1, TPR at fixed FPR, hour-matched and volume-stratified AUC, recall per attack type |
| `lodo_summary.py` | Pooled and within-day metrics over the LODO folds |
| `run_all.py` | Runs all experiments as a resumable job queue |
| `analysis/paper_analysis.py` | Secondary analyses: configuration selection, recall per attack type under LODO, fold groups, paired block bootstrap, Attempted-only windows |
| `splits/` | Reference copies of the window table and of all 5-minute splits (see below) |

Labels are read only by `merge_build.py`, which writes the label table and the splits, and by `evaluate.py`,
`lodo_summary.py` and the analysis script. The detectors never read labels. The only exception is the flow-level
baselines, which use the flow labels to sample benign flows for training.

## Installation

Python 3.11 or 3.12. A CUDA GPU is recommended; every script also runs on CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the corrected CIC-IDS2017 and CSE-CIC-IDS2018 CSV files (Engelen et al., 2021; Liu et al., 2022) from
https://intrusion-detection.distrinet-research.be/CNS2022/. Put all files in one folder, named as in `datasets.py`:
- CIC-IDS2017: `monday.csv` ... `friday.csv`;
- CSE-CIC-IDS2018: `Wednesday-14-02-2018.csv` ... `Friday-02-03-2018.csv`.

## Running

Quick check of the installation (two CIC-IDS2017 days, two epochs, a few minutes):

```bash
python run_all.py --csv-dir <csv_dir> --root runs_smoke --smoke --gpus 0 --stages day lodo
```

All experiments of the paper:

```bash
python run_all.py --csv-dir <csv_dir> --root runs --gpus <number of GPUs> --jobs-per-gpu 2 --threads 8
python analysis/paper_analysis.py --root runs --out analysis_out
```

The stages can also be run separately with `--stages day`, `--stages lodo` and `--stages windows`.
Finished jobs are skipped, so an interrupted run resumes with the same command.
On one A100 GPU, one GRPD run on CSE-CIC-IDS2018 takes about 7 minutes. The flow-level baselines are the slowest jobs
and run on CPU.

Every detector can also be run on its own, for example:

```bash
python grpd.py --data runs/cic2018/graphs/data --split runs/cic2018/graphs/data/split_day.csv --out my_run --seed 42
python evaluate.py runs/cic2018/graphs/data <folder with runs> <output folder> --main grpd_s42
```

## Outputs and the paper

| Paper | Output |
|---|---|
| Day-disjoint results (5-min windows) | `runs/<ds>/eval/day/metrics.csv`, `metrics_by_seed.csv` |
| LODO results | `runs/<ds>/eval/lodo/pooled.csv`, `pooled_by_seed.csv`, `per_fold.csv` |
| LODO by fold group | `analysis_out/fold_groups.json` |
| Recall per attack type (LODO) | `analysis_out/per_attack_<ds>.csv` |
| Random within-day split | runs `grpd_random_s*`, `bl_graph_random` and `bl_flow_random` in `runs/<ds>/eval/day/metrics.csv` |
| 1-minute windows and window-length sweep | `runs/<ds>/eval/w<k>/metrics.csv` |
| Fixed configuration rule vs. best test configuration | `analysis_out/config_selection.csv` |
| Message passing (MLP encoder) | runs `grpd_mlp_s*` in `runs/<ds>/eval/day/metrics_by_seed.csv` |
| Paired block-bootstrap intervals, Attempted-only windows | `analysis_out/paired_bootstrap.json`, `attempted_2017.json` |

Score columns:
- GRPD: `S_grpd`.
- Anomal-E: `ANOMALE_<detector>_<aggregate>`.
- PCA control: `PCAREC_REC_<aggregate>`.
- DOMINANT and OCGNN: `DOMINANT_<aggregate>`, `OCGNN_<aggregate>`.
- Window-vector and flow-level baselines: `G188_*`, `G47_*`, `FLOW_*`.
- Flow count: `n_flows` (run `confound`).

## Splits and window labels

`splits/<dataset>/windows.csv` lists every window that has at least 20 nodes and 20 edges. For each window it gives:
- graph id, source file, start and end time;
- numbers of nodes, edges and flows;
- label columns: `has_attack_strict`, `attack_type`, numbers of attack and *Attempted* flows.

The `split_*.csv` files give the role of each window:
- **FIT**: training;
- **CAL**: early stopping;
- **VAL**: threshold and empirical CDF references;
- **TEST**: test;
- **TEST_NEAR**: near-attack benign windows, reported separately;
- **TEST_ATTEMPTED**: windows whose only non-benign flows are *Attempted*, excluded from the metrics;
- **TRAINDAY_ATTACK**: attack windows on non-test days, not used;
- **OTHER**: not used.

A benign window is *safe* if no window within 15 minutes on the same day contains an attack or *Attempted* flow.
`run_all.py` regenerates the same files deterministically; the copies here are for reference and checking.

## Notes

- Seeds: GRPD uses seeds 42–46 on the day-disjoint split and 42–44 elsewhere. Anomal-E, DOMINANT and OCGNN use
  seeds 42–44.
- GRPD uses equal weights in every fusion and library defaults for the optimizer, the isolation forest and the
  scalers (`CFG` in `grpd.py`).
- For the multi-configuration baselines, the tables of the paper report the configuration with the best test AUC.
  `analysis/paper_analysis.py` also reports one fixed rule (isolation forest, top-10 mean) and the median over
  configurations.
- Results on GPU are not bit-identical across hardware. Expect differences in the third decimal.

## License

MIT (to be confirmed at publication).
