"""Leave-one-day-out (LODO) splits: every capture day that contains attacks is the test day once.

Protocol (fixed before any LODO run; secondary to the chronological split in datasets.py):
  TEST day d: every strict-attack window + safe benign windows -> TEST; pure benign windows that fail the safe gap
              -> TEST_NEAR; windows with only 'Attempted' non-benign flows -> TEST_ATTEMPTED.
  All other days: safe benign windows are split by whole clock-hour blocks inside each day (counted from the first
              window of the day): block % 4 == 3 -> VAL (ECDF references, the Q85 threshold and AE early stopping),
              all other blocks -> FIT. FIT windows next to a VAL window (+-1 window, same day) are dropped (OTHER) so
              that FIT never touches VAL in time. Strict-attack windows of these days -> TRAINDAY_ATTACK (never used).
Usage: python make_lodo.py <data_dir>      -> <data_dir>/split_lodo_<day>.csv for every day with attacks + lodo_summary.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

SAFE_GAP = 3


def main():
    data = Path(sys.argv[1])
    L = pd.read_parquet(data / 'labels.parquet')
    L['win_start'] = pd.to_datetime(L.win_start)
    pure = (L.has_attack_any.eq(0) & L.raw_attempted_flows.eq(0)).to_numpy()
    bad = {(s, t) for s, t in zip(L.source_file[~pure], L.win_start[~pure])}
    step = pd.Timedelta(minutes=5)
    safe = np.array([bool(p) and all((s, t + k * step) not in bad for k in range(-SAFE_GAP, SAFE_GAP + 1))
                     for p, s, t in zip(pure, L.source_file, L.win_start)])
    strict = L.has_attack_strict.eq(1).to_numpy()
    # hour blocks per day
    first = L.groupby('source_file').win_start.transform('min')
    block = ((L.win_start - first.dt.floor('h')).dt.total_seconds() // 3600).astype(int).to_numpy()
    base = np.where(block % 4 == 3, 'VAL', 'FIT')
    summary = {}
    for day in sorted(L.source_file.unique()):
        dm = (L.source_file == day).to_numpy()
        if not strict[dm].any():
            continue
        role = np.full(len(L), 'OTHER', dtype=object)
        role[~dm & safe] = base[~dm & safe]
        role[~dm & strict] = 'TRAINDAY_ATTACK'
        # guard: drop FIT windows adjacent (+-1 window, same day) to a VAL window
        key = {(s, t): r for s, t, r in zip(L.source_file, L.win_start, role)}
        for i in np.flatnonzero(role == 'FIT'):
            s, t = L.source_file.iat[i], L.win_start.iat[i]
            if any(key.get((s, t + k * step)) == 'VAL' for k in (-1, 1)):
                role[i] = 'OTHER'
        role[dm & (strict | safe)] = 'TEST'
        role[dm & pure & ~safe] = 'TEST_NEAR'
        role[dm & ~pure & ~strict] = 'TEST_ATTEMPTED'
        stem = Path(day).stem
        pd.DataFrame(dict(graph_id=L.graph_id, role=role)).to_csv(data / f'split_lodo_{stem}.csv', index=False)
        summary[stem] = dict(pd.Series(role).value_counts().to_dict(), test_attack=int((dm & strict).sum()),
                             test_benign=int((dm & safe & ~strict).sum()))
    (data / 'lodo_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
