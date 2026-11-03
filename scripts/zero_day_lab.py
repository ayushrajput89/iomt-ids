"""Zero-day lab from the command line (same logic as app/zero_day_lab.py, with a simulated analyst).

  python scripts/zero_day_lab.py                                   # all 8 scenarios, summary table
  python scripts/zero_day_lab.py --scenario "ARP spoofing" --mode no-replay
Simulated analyst: a pattern that is >= 80% unseen-attack windows is named after its majority unseen class (several
patterns may get the same name); other patterns are dismissed as false alarms. An upper bound on analyst quality.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from iomt_ids.zero_day import ZeroDayLab, scenarios   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--scenario', default='all')
ap.add_argument('--mode', default='replay', choices=['replay', 'no-replay', 'retrain'])
ap.add_argument('--method', default='knn+energy', choices=['knn+energy', 'knn', 'energy'])
ap.add_argument('--fpr', type=float, default=0.05)
ap.add_argument('--max-windows', type=int, default=1000, help='labelled windows used per new class')
ap.add_argument('--include-alerts', action='store_true', help='also group windows alerted as a known attack')
a = ap.parse_args()

zd = ROOT / 'data' / 'zeroday'
if not (zd / 'reference_train.parquet').exists():
    sys.exit('data/zeroday/reference_train.parquet not found - run notebooks/v13_zero_day_continual.py and copy its exports.')
cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
ref, val = pd.read_parquet(zd / 'reference_train.parquet'), pd.read_parquet(zd / 'reference_val.parquet')
stream = pd.read_parquet(ROOT / cfg['demo_stream'])
feats = [c for c in ref.columns if c != 'label' and c in stream.columns]; labels = sorted(ref['label'].unique())
SC = scenarios(labels)
todo = SC if a.scenario == 'all' else {a.scenario: SC[a.scenario]}

rows = []
for name, hidden in todo.items():
    lab = ZeroDayLab(ref, val, stream, feats, labels, hidden, cache_dir=ROOT / 'models' / 'lab' / 'cache', method=a.method, target_fpr=a.fpr)
    b, _ = lab.train_base(); lab.fit_novelty(b); S = lab.score(b)
    s0, e0 = lab.summary(S, 'evaluation'), lab.evaluate(b)
    learned = {}
    idx, plab, summ, _ = lab.discover(S, include_alerts=a.include_alerts)
    for _, p in summ.iterrows():                                         # simulated analyst reviews every pattern
        win = idx[plab == p['id']]; tr = pd.Series(lab.true[win])
        if tr.isin(hidden).mean() < 0.8: continue                         # mostly known/benign traffic -> dismissed
        name_ = tr[tr.isin(hidden)].value_counts().index[0]               # analyst names the new attack
        learned.setdefault(name_, []).append(win)
    for name_, wins in learned.items():
        win = np.concatenate(wins)[:a.max_windows]
        b, info = lab.learn(b, win, name_, a.mode)
        print(f'  [{name}] learned {name_} from {len(win)} windows in {len(wins)} pattern(s) ({info["mode"]}, {info["train_sec"]} s)')
    S = lab.score(b)
    s1, e1 = lab.summary(S, 'evaluation'), lab.evaluate(b)
    rows.append({'scenario': name, 'learned': ', '.join(learned) or '-', 'caught %': s0['unseen_caught_%'],
                 'unknown %': s0['unseen_flagged_unknown_%'], 'benign false-unknown %': s0['benign_false_unknown_%'],
                 'unseen recognized % (after)': e1['unseen_recognized_%'], 'unseen F1 after': e1['new_macro_f1'],
                 'old F1 before': e0['old_macro_f1'], 'old F1 after': e1['old_macro_f1']})
    print(f'{name}: done')

pd.set_option('display.width', 220)
print('\nZero-day lab - evaluation half of the demo stream (segments never labelled):')
print(pd.DataFrame(rows).round(1).to_string(index=False))
