"""v14 - Is window size a second shortcut?

CICIoMT2024 uses two window sizes: low-rate classes are averaged over small windows and flood classes over
large ones (k_window = 10 vs 20 in the IAT decoding). If a model used window size as a cue, it could only
separate these two groups of classes. This check measures how much that could explain.

Inputs : results_v10a/probs_seed42.npz   (y_static, P_static, labels - shortcut-free XGBoost, seed 42)
         data/iat_decode_ciciomt2024.csv  (k_window per class)
Output : printed table + results_v14/window_size_check.csv
Runs on CPU in a few seconds.

usage: python v14_window_size_check.py <probs_seed42.npz> <iat_decode_ciciomt2024.csv> [out_dir]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

npz, decode = sys.argv[1], sys.argv[2]
out_dir = sys.argv[3] if len(sys.argv) > 3 else 'results_v14'
os.makedirs(out_dir, exist_ok=True)

k = pd.read_csv(decode).groupby('label')['k_window'].first()
z = np.load(npz, allow_pickle=True)
labels = [str(l) for l in z['labels']]
y, P = z['y_static'], z['P_static']
pred = P.argmax(1)
group = np.array([k[l] for l in labels])          # window-size group of every class
gy, gp = group[y], group[pred]

rows = [{'check': 'overall accuracy (%)', 'value': (pred == y).mean() * 100},
        {'check': 'predictions that cross window-size groups (% of windows)', 'value': (gy != gp).mean() * 100},
        {'check': 'errors that cross window-size groups (% of errors)', 'value': (gy != gp)[pred != y].mean() * 100}]

# Upper bound for a classifier that knows ONLY the window-size group: predict the most frequent test class of the group
oracle = np.empty_like(y)
for g in np.unique(group):
    m = gy == g
    oracle[m] = np.bincount(y[m]).argmax()
rows += [{'check': 'window-size-only oracle: accuracy (%)', 'value': (oracle == y).mean() * 100},
         {'check': 'window-size-only oracle: macro-F1 (%)', 'value': f1_score(y, oracle, average='macro') * 100}]

# Within each group: accuracy of the model vs the same model restricted to the classes of the true group
for g in np.unique(group):
    cls = np.flatnonzero(group == g)
    m = np.isin(y, cls)
    restricted = cls[P[m][:, cls].argmax(1)]
    rows += [{'check': f'group k={g} ({len(cls)} classes): accuracy (%)', 'value': (pred[m] == y[m]).mean() * 100},
             {'check': f'group k={g}: accuracy when the group is given (%)', 'value': (restricted == y[m]).mean() * 100}]

T = pd.DataFrame(rows).round(2)
T.to_csv(os.path.join(out_dir, 'window_size_check.csv'), index=False)
print(T.to_string(index=False))
print('\nGroups:')
for g in np.unique(group):
    print(f'  k={g}:', ', '.join(l for l in labels if k[l] == g))
