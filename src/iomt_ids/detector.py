"""M4 - Context-Aware Detector: p = w * P(per-window XGB) + (1 - w) * P(context XGB)."""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


class ContextAwareDetector:
    def __init__(self, models_dir, labels, fusion_w, nthreads=1):
        models_dir = Path(models_dir)
        self.window_model = xgb.Booster(); self.window_model.load_model(str(models_dir / 'm_window.json'))
        self.context_model = xgb.Booster(); self.context_model.load_model(str(models_dir / 'm_context.json'))
        for b in (self.window_model, self.context_model):
            b.set_param({'device': 'cpu', 'nthread': nthreads})
        self.labels = list(labels)
        self.w = float(fusion_w)

    def predict_proba_one(self, x, ctx):
        pa = self.window_model.inplace_predict(x[None, :])
        pc = self.context_model.inplace_predict(ctx[None, :])
        return (self.w * pa + (1 - self.w) * pc)[0]

    def predict_proba_batch(self, X, k):
        """Vectorised version over an ordered stream (same result as streaming)."""
        roll = pd.DataFrame(X).rolling(k, min_periods=1)
        ctx = np.hstack([X, roll.mean().values, roll.std().fillna(0).values]).astype('float32')
        return self.w * self.window_model.inplace_predict(X) + (1 - self.w) * self.context_model.inplace_predict(ctx)

    def set_threads(self, n):
        for b in (self.window_model, self.context_model):
            b.set_param({'nthread': n})
