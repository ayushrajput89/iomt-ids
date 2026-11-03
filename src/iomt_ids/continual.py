"""M10 - Continual learning: the analyst labels a discovered pattern, the model learns it without forgetting.

The label space has fixed slots (all class names the deployment may ever report). A class that was never seen
simply has no training rows yet, so the booster can be FINE-TUNED (continued boosting) when it is labelled:
  * 'replay'   - continued boosting on new windows + a replay buffer of old classes (default; seconds)
  * 'no-replay'- continued boosting on new windows only (shown for comparison: catastrophic forgetting)
  * 'retrain'  - train from scratch on the replay buffer + new windows
A brand-new name that is not in the slot list is appended and forces 'retrain'.
Measured on the official test set in v13 / Z4.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

MODES = ('replay', 'no-replay', 'retrain')


def base_params(n_classes, nthread=-1, seed=42, depth=6):
    return {'objective': 'multi:softprob', 'num_class': n_classes, 'eta': 0.1, 'max_depth': depth, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'tree_method': 'hist', 'device': 'cpu', 'nthread': nthread, 'seed': seed,
            'eval_metric': 'mlogloss'}


# A class absent from base training has P ~ 0, so its hessian ~ 0 and default settings never grow its trees.
FINETUNE_OVERRIDE = {'eta': 0.3, 'lambda': 1e-3, 'min_child_weight': 0, 'max_delta_step': 2}


def _dmat(X, y):
    return xgb.DMatrix(np.asarray(X, dtype='float32'), label=y, weight=compute_sample_weight('balanced', y))


def train_booster(X, y, n_classes, rounds=150, nthread=-1, seed=42):
    """From scratch, with 15% early-stopping split."""
    u = np.random.default_rng(seed).random(len(y)); a, b = u < 0.85, u >= 0.85
    return xgb.train(base_params(n_classes, nthread, seed), _dmat(X[a], y[a]), rounds,
                     evals=[(_dmat(X[b], y[b]), 'val')], early_stopping_rounds=20, verbose_eval=False)


def finetune_booster(base, X, y, n_classes, rounds=60, nthread=-1, seed=42):
    p = base_params(n_classes, nthread, seed); p.update(FINETUNE_OVERRIDE)
    return xgb.train(p, _dmat(X, y), rounds, xgb_model=base, verbose_eval=False)


def evaluate(booster, X, y, labels, known=None, new=None):
    """Macro-F1 over old (known) classes, new classes and all classes present in y (label indices)."""
    p = booster.inplace_predict(np.asarray(X, dtype='float32')).argmax(1); out = {'accuracy_%': float((p == y).mean() * 100)}
    present = sorted(set(np.asarray(y).tolist()))
    for name, cl in (('old_macro_f1', known), ('new_macro_f1', new), ('all_macro_f1', present)):
        cl = [c for c in (cl or []) if c in present]
        out[name] = float(f1_score(y, p, labels=cl, average='macro', zero_division=0) * 100) if cl else None
    if new:
        m = np.isin(y, new); out['new_recall_%'] = float((p[m] == y[m]).mean() * 100) if m.any() else None
    out['per_class_f1'] = dict(zip([labels[c] for c in present],
                                   (f1_score(y, p, labels=present, average=None, zero_division=0) * 100).round(2).tolist()))
    return out


class ContinualLearner:
    def __init__(self, labels, replay_X, replay_y, replay_per_class=500, nthread=-1, seed=42):
        """labels: slot names; replay_X/replay_y: reference windows of KNOWN classes (label indices)."""
        self.labels = list(labels); self.nthread, self.seed = nthread, seed
        self.replay_X, self.replay_y = np.asarray(replay_X, dtype='float32'), np.asarray(replay_y)
        self.replay_per_class = replay_per_class; self.history = []

    def label_index(self, name):
        if name not in self.labels: self.labels.append(name)
        return self.labels.index(name)

    def _replay(self, per_class):
        r = np.random.default_rng(self.seed); idx = []
        for c in np.unique(self.replay_y):
            w = np.flatnonzero(self.replay_y == c); idx.extend(r.choice(w, min(per_class, len(w)), replace=False))
        idx = np.array(sorted(idx)); return self.replay_X[idx], self.replay_y[idx]

    def learn(self, booster, X_new, name, mode='replay', rounds=60):
        """Returns (new booster, info). The new windows are added to the replay buffer for future updates."""
        if mode not in MODES: raise ValueError(f'mode must be one of {MODES}')
        n_before = len(self.labels); c = self.label_index(name); y_new = np.full(len(X_new), c)
        if len(self.labels) != n_before and mode != 'retrain':
            mode = 'retrain'                                   # new slot -> booster shape changes
        t = time.perf_counter()
        if mode == 'no-replay':
            bst = finetune_booster(booster, X_new, y_new, len(self.labels), rounds, self.nthread, self.seed)
        elif mode == 'replay':
            Xr, yr = self._replay(self.replay_per_class)
            bst = finetune_booster(booster, np.vstack([Xr, X_new]), np.r_[yr, y_new], len(self.labels), rounds, self.nthread, self.seed)
        else:
            bst = train_booster(np.vstack([self.replay_X, X_new]), np.r_[self.replay_y, y_new], len(self.labels),
                                nthread=self.nthread, seed=self.seed)
        sec = time.perf_counter() - t
        self.replay_X = np.vstack([self.replay_X, np.asarray(X_new, dtype='float32')]); self.replay_y = np.r_[self.replay_y, y_new]
        info = {'label': name, 'slot': c, 'mode': mode, 'n_windows': int(len(X_new)), 'train_sec': round(sec, 2),
                'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        self.history.append(info); return bst, info


class ModelRegistry:
    """models/lab/: every learned model is saved as a new version with its lineage and metrics."""
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True); self.index = self.root / 'registry.json'

    def entries(self):
        return json.loads(self.index.read_text()) if self.index.exists() else []

    def save(self, booster, labels, info, metrics=None, parent=None):
        e = self.entries(); v = f'v{len(e) + 1}'; booster.save_model(str(self.root / f'{v}.json'))
        e.append({'version': v, 'parent': parent, 'labels': list(labels), **info, 'metrics': metrics or {}})
        self.index.write_text(json.dumps(e, indent=1, default=float)); return v

    def table(self):
        rows = [{k: v for k, v in r.items() if k not in ('labels', 'metrics')} | {k: v for k, v in (r.get('metrics') or {}).items()
                if not isinstance(v, dict)} for r in self.entries()]
        return pd.DataFrame(rows)
