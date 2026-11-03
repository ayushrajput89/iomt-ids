"""M11 - Drift monitor: has the traffic distribution moved away from what the model was trained on?

Population Stability Index (PSI) per feature, bins = reference deciles (in the signed-log space).
Rule of thumb: PSI < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 major shift (retraining / review advised).
Raw CIC CSVs are also checked for the capture-timestamp artifact in IAT (module audit, M0).
"""
import numpy as np
import pandas as pd

from .audit import decode_iat_values

LEVELS = [(0.10, 'stable'), (0.25, 'moderate shift'), (np.inf, 'major shift')]


def _t(x):
    x = np.asarray(x, dtype='float64'); return np.sign(x) * np.log1p(np.abs(x))


def psi(ref, cur, bins=10, eps=1e-4):
    r, c = _t(ref), _t(cur)
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # (near) constant feature: compare share of the main value
        v = np.median(r); pr, pc = np.mean(np.isclose(r, v)), np.mean(np.isclose(c, v))
        P, Q = np.array([pr, 1 - pr]), np.array([pc, 1 - pc])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        P = np.histogram(r, edges)[0] / len(r); Q = np.histogram(c, edges)[0] / len(c)
    P, Q = np.clip(P, eps, None), np.clip(Q, eps, None)
    return float(np.sum((Q - P) * np.log(Q / P)))


def level(v):
    return next(name for lim, name in LEVELS if v < lim)


def drift_report(ref_df, cur_df, features):
    rows = [{'feature': f, 'PSI': psi(ref_df[f].to_numpy(), cur_df[f].to_numpy())} for f in features if f in cur_df]
    out = pd.DataFrame(rows).sort_values('PSI', ascending=False).reset_index(drop=True)
    out['status'] = out['PSI'].map(level); return out


def summarize(report):
    n = report['status'].value_counts()
    worst = report.iloc[0] if len(report) else None
    overall = 'major shift' if n.get('major shift', 0) >= 3 else 'moderate shift' if n.get('major shift', 0) + n.get('moderate shift', 0) >= 3 else 'stable'
    return {'overall': overall, 'features_major': int(n.get('major shift', 0)), 'features_moderate': int(n.get('moderate shift', 0)),
            'features_stable': int(n.get('stable', 0)), 'worst_feature': None if worst is None else worst['feature'],
            'worst_psi': None if worst is None else float(worst['PSI'])}


def check_raw_capture(df):
    """For a raw CIC-style CSV: is IAT a capture timestamp? (then it must never reach a model)."""
    col = next((c for c in df.columns if str(c).strip().lower() == 'iat'), None)
    return None if col is None else decode_iat_values(df[col].to_numpy())
