"""M7 - Explainer: TreeSHAP (XGBoost built-in, exact) for the per-window and the context model.

- explain(x, label)            -> top-k reasons for one alert (used in alerts)
- explain_full(x, label, ctx)  -> all contributions + base value (waterfall chart)
- global_importance(X, ...)    -> mean |SHAP| per feature, overall and per class
"""
import numpy as np
import pandas as pd
import xgboost as xgb


def context_feature_names(features):
    return list(features) + [f'mean10_{f}' for f in features] + [f'std10_{f}' for f in features]


class Explainer:
    def __init__(self, booster, features, labels, top_k=3, context_booster=None):
        self.booster, self.features, self.labels, self.k = booster, list(features), list(labels), top_k
        self.ctx_booster = context_booster
        self.ctx_features = context_feature_names(features)

    @staticmethod
    def _contribs(booster, X):
        c = booster.predict(xgb.DMatrix(np.asarray(X, dtype='float32')), pred_contribs=True)
        return c if c.ndim == 3 else c[:, None, :]            # (n, classes, f+1); last column = bias

    def explain(self, x, label):
        cls = self.labels.index(label)
        v = self._contribs(self.booster, np.asarray(x)[None, :])[0, cls, :-1]
        idx = np.argsort(-np.abs(v))[:self.k]
        return [f"{self.features[i]} ({'+' if v[i] > 0 else '-'}{abs(v[i]):.2f})" for i in idx]

    def explain_full(self, x, label, ctx=None, model='window'):
        """Returns DataFrame(feature, value, contribution) sorted by |contribution| and the base value (log-odds)."""
        cls = self.labels.index(label)
        if model == 'context':
            if self.ctx_booster is None or ctx is None:
                raise ValueError('context model or context vector missing')
            b, vec, names = self.ctx_booster, np.asarray(ctx), self.ctx_features
        else:
            b, vec, names = self.booster, np.asarray(x), self.features
        c = self._contribs(b, vec[None, :])[0, cls]
        df = pd.DataFrame({'feature': names, 'value': vec, 'contribution': c[:-1]})
        df = df.reindex(df['contribution'].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return df, float(c[-1])

    def global_importance(self, X, y_labels=None, model='window', max_rows=3000, seed=0):
        """Mean |SHAP| per feature. Overall = summed over classes; per class = contribution to the true class."""
        X = np.asarray(X, dtype='float32')
        if len(X) > max_rows:
            idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False); X = X[idx]
            y_labels = None if y_labels is None else np.asarray(y_labels)[idx]
        b, names = (self.ctx_booster, self.ctx_features) if model == 'context' else (self.booster, self.features)
        C = self._contribs(b, X)[:, :, :-1]
        overall = pd.Series(np.abs(C).sum(1).mean(0), index=names).sort_values(ascending=False)
        per_class = None
        if y_labels is not None:
            yi = np.array([self.labels.index(l) for l in y_labels])
            own = np.abs(C[np.arange(len(yi)), yi, :])
            per_class = pd.DataFrame(own, columns=names).assign(label=np.asarray(y_labels)).groupby('label').mean()
        return overall, per_class
