"""M9 - Pattern discovery: groups unknown windows into candidate NEW attack patterns for an analyst.

KMeans in the novelty feature space, k chosen by silhouette (k = 1 when no clear structure). Each pattern gets an
interpretable profile: how far each feature is from normal (Benign) traffic, and which known class is nearest.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def choose_k(Z, k_max=6, min_sil=0.25, seed=0, max_rows=3000):
    if len(Z) < 40: return 1, {}
    sub = Z[np.random.default_rng(seed).choice(len(Z), min(len(Z), max_rows), replace=False)]
    sil = {}
    for k in range(2, min(k_max, len(sub) - 1) + 1):
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(sub)
        if len(set(lab)) > 1: sil[k] = float(silhouette_score(sub, lab))
    if not sil or max(sil.values()) < min_sil: return 1, sil
    return max(sil, key=sil.get), sil


def discover_patterns(X, space, ref_X, ref_y, features, k=None, k_max=6, min_size=15, seed=0, benign='Benign'):
    """X: unknown windows (raw features). space: novelty FeatureSpace. ref_X/ref_y: reference set (known classes).
    Returns (pattern id per window, -1 = too small to report; summary DataFrame; silhouette per k)."""
    Z = space(X); sil = {}
    if k is None: k, sil = choose_k(Z, k_max, seed=seed)
    k = max(1, min(int(k), len(Z) // max(min_size, 1)))
    lab = KMeans(k, n_init=5, random_state=seed).fit_predict(Z) if k > 1 else np.zeros(len(Z), int)
    sizes = pd.Series(lab).value_counts()
    small = sizes[sizes < min_size].index; lab = np.where(np.isin(lab, small), -1, lab)
    order = {old: new for new, old in enumerate(sizes[sizes >= min_size].index)}       # P1 = largest
    lab = np.array([order.get(l, -1) for l in lab])

    ref_y = np.asarray(ref_y); Zr = space(ref_X)
    cls = sorted(set(ref_y)); cent = np.vstack([Zr[ref_y == c].mean(0) for c in cls])
    zb = Zr[ref_y == benign] if benign in cls else Zr
    b_mu, b_sd = zb.mean(0), zb.std(0) + 0.25                                           # floor: avoid huge z on constant features
    rows = []
    for p in sorted(set(lab) - {-1}):
        Zp = Z[lab == p]; c = Zp.mean(0)
        d = np.linalg.norm(cent - c, axis=1); near = np.argsort(d)[:2]
        z = (np.median(Zp, 0) - b_mu) / b_sd; top = np.argsort(-np.abs(z))[:4]
        rows.append({'pattern': f'P{p + 1}', 'id': int(p), 'windows': int(len(Zp)),
                     'nearest_known': cls[near[0]], 'second_nearest': cls[near[1]] if len(near) > 1 else '',
                     'distance_to_nearest': float(d[near[0]]), 'spread': float(np.linalg.norm(Zp - c, axis=1).mean()),
                     'signature': ', '.join(f"{features[i]} {'↑' if z[i] > 0 else '↓'}{abs(z[i]):.1f}σ" for i in top)})
    return lab, pd.DataFrame(rows), sil


def project_2d(Z, seed=0):
    """PCA to 2-D for plotting (numpy SVD, no extra dependency)."""
    Zc = Z - Z.mean(0); _, _, vt = np.linalg.svd(Zc[np.random.default_rng(seed).choice(len(Zc), min(len(Zc), 5000), replace=False)],
                                                full_matrices=False)
    return Zc @ vt[:2].T
