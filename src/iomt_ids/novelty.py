"""M8 - Novelty (open-set) detector: flags windows that do not look like ANY class the model was trained on.

Two complementary scores, both calibrated on KNOWN validation windows only:
  * kNN distance  - mean distance to the k nearest reference windows in a signed-log, z-scored feature space
                    (model-free: catches traffic that is far from everything seen in training)
  * energy        - -logsumexp of the classifier margins over the known classes
                    (model-based: catches windows on which no known class is confident)
Each score is turned into an empirical percentile against the validation scores; the combined score is the max
of the two percentiles. A window is UNKNOWN when its score exceeds the (1 - target_fpr) validation quantile,
so ~target_fpr of known traffic is expected to be flagged (measured on the official test set in v13 / Z1).
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

METHODS = ('knn', 'energy', 'knn+energy')


class FeatureSpace:
    """Signed log1p + z-score. CIC features span 8+ orders of magnitude, so raw Euclidean distance is meaningless."""
    def __init__(self, X_ref):
        Z = self._t(X_ref); self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6

    @staticmethod
    def _t(X):
        X = np.asarray(X, dtype='float64'); return np.sign(X) * np.log1p(np.abs(X))

    def __call__(self, X):
        return ((self._t(X) - self.mu) / self.sd).astype('float32')


def energy_score(margins, known_idx=None):
    m = np.asarray(margins, dtype='float64')
    if known_idx is not None: m = m[:, known_idx]
    return -logsumexp(m, axis=1)


class NoveltyDetector:
    def __init__(self, k=10, target_fpr=0.05, method='knn+energy'):
        if method not in METHODS: raise ValueError(f'method must be one of {METHODS}')
        self.k, self.target_fpr, self.method = k, target_fpr, method

    def fit(self, X_ref, X_val, margins_val=None, known_idx=None):
        """X_ref: reference windows of known classes; X_val: held-out known windows used ONLY for thresholds."""
        if self.method != 'knn' and margins_val is None:
            raise ValueError('energy-based methods need classifier margins for the validation windows')
        self.space = FeatureSpace(X_ref); self.known_idx = None if known_idx is None else list(known_idx)
        self._ref = self.space(X_ref)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(self._ref))).fit(self._ref)
        self.val_knn = np.sort(self._knn(X_val))
        self.val_energy = None if margins_val is None else np.sort(energy_score(margins_val, self.known_idx))
        self.threshold = float(np.quantile(self._combine(self._knn(X_val), margins_val), 1 - self.target_fpr))
        return self

    def add_reference(self, X_new, known_idx=None):
        """After the model learns a new class, its windows join the reference set (then call recalibrate)."""
        self._ref = np.vstack([self._ref, self.space(X_new)])
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(self._ref))).fit(self._ref)
        if known_idx is not None: self.known_idx = list(known_idx)
        return self

    def recalibrate(self, X_val, margins_val=None):
        """Re-derive validation percentiles + threshold after the classifier changed (margins shift after fine-tuning)."""
        self.val_knn = np.sort(self._knn(X_val))
        if margins_val is not None: self.val_energy = np.sort(energy_score(margins_val, self.known_idx))
        self.threshold = float(np.quantile(self._combine(self._knn(X_val), margins_val), 1 - self.target_fpr))
        return self

    def _knn(self, X):
        return self.nn.kneighbors(self.space(X))[0].mean(1)

    @staticmethod
    def _pct(sorted_ref, s):
        return np.searchsorted(sorted_ref, s, side='right') / len(sorted_ref)

    def _combine(self, knn, margins):
        if self.method == 'knn': return self._pct(self.val_knn, knn)
        e = energy_score(margins, self.known_idx)
        if self.method == 'energy': return self._pct(self.val_energy, e)
        return np.maximum(self._pct(self.val_knn, knn), self._pct(self.val_energy, e))

    def score(self, X, margins=None):
        """Returns (novelty score in [0, 1], unknown flag, raw kNN distance)."""
        knn = self._knn(X); s = self._combine(knn, margins)
        return s, s > self.threshold, knn
