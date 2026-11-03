"""Zero-day lab: simulate an attack the deployed model has never seen, detect it, group it, label it, learn it.

  1. hide an attack group  -> train a window-level lab model on the reference set WITHOUT it (19 label slots kept)
  2. replay the demo stream -> each window: known-attack alert (P(attack) >= threshold), UNKNOWN pattern (novelty
     sustained in 3 of the last 5 windows), or pass
  3. discovery half of the stream -> unknown windows are grouped into candidate patterns (M9)
  4. analyst names a pattern -> continual learning with replay (M10); novelty reference grows (M8)
  5. evaluation half (segments never used for labelling) -> before/after: new-class F1, old-class F1 (forgetting)
The lab model sees single windows only (no context model), so its absolute scores are below the deployed ensemble.
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .continual import ContinualLearner, evaluate, train_booster
from .novelty import NoveltyDetector
from .patterns import discover_patterns

SCENARIO_RULES = {
    'ARP spoofing': lambda l: l == 'ARP_Spoofing',
    'Reconnaissance (4 types)': lambda l: l.startswith('Recon'),
    'MQTT attacks (5 types)': lambda l: l.startswith('MQTT'),
    'MQTT malformed data': lambda l: l == 'MQTT-Malformed_Data',
    'ICMP floods (DoS + DDoS)': lambda l: l.endswith('-ICMP'),
    'SYN floods (DoS + DDoS)': lambda l: l.endswith('-SYN'),
    'UDP floods (DoS + DDoS)': lambda l: l.endswith('-UDP'),
    'TCP floods (DoS + DDoS)': lambda l: l.endswith('-TCP'),
}


def scenarios(labels):
    out = {k: [l for l in labels if f(l)] for k, f in SCENARIO_RULES.items()}
    return {k: v for k, v in out.items() if v}


def split_discovery(stream, frac=0.5, seed=0):
    """True = discovery half (analyst may label from it), False = evaluation half. Split by whole segments."""
    seg = stream['segment'].to_numpy() if 'segment' in stream else np.arange(len(stream)) // 100
    lab = stream['label'].to_numpy(); r = np.random.default_rng(seed); disc = set()
    for l in np.unique(lab):                                             # stratified: every class in both halves
        s = np.unique(seg[lab == l]); r.shuffle(s); disc.update(s[:max(1, int(round(frac * len(s))))] if len(s) > 1 else s)
    m = np.isin(seg, list(disc))
    for l in np.unique(lab):                                             # single-segment class: split it in time
        w = np.flatnonzero(lab == l)
        if m[w].all() or (~m[w]).all(): m[w] = np.arange(len(w)) < len(w) * frac
    return m


class ZeroDayLab:
    def __init__(self, ref, val, stream, features, labels, hidden, cache_dir=None, nthread=-1,
                 method='knn+energy', target_fpr=0.05, attack_threshold=0.95, unknown_smoothing=(3, 5), seed=0):
        self.features, self.labels, self.hidden = list(features), list(labels), list(hidden)
        self.L2I = {l: i for i, l in enumerate(self.labels)}; self.benign = self.L2I['Benign']
        self.known = [self.L2I[l] for l in self.labels if l not in self.hidden]
        self.learned = []                                                # slots learned in this session
        kref, kval = ref[~ref['label'].isin(hidden)], val[~val['label'].isin(hidden)]
        self.Xr, self.yr = kref[self.features].to_numpy('float32'), kref['label'].map(self.L2I).to_numpy()
        self.Xv = kval[self.features].to_numpy('float32')
        self.Xs = stream[self.features].to_numpy('float32'); self.true = stream['label'].to_numpy()
        self.ys = np.array([self.L2I.get(l, -1) for l in self.true])
        self.disc = split_discovery(stream, seed=seed)
        self.cache_dir, self.nthread, self.thr, self.seed = cache_dir, nthread, attack_threshold, seed
        self.method, self.target_fpr, self.smoothing = method, target_fpr, unknown_smoothing
        self.learner = ContinualLearner(self.labels, self.Xr, self.yr, nthread=nthread)

    # ------------------------------------------------------------------ base model + novelty
    def _cache_path(self):
        if not self.cache_dir: return None
        key = hashlib.md5(('|'.join(self.features) + '#' + '|'.join(self.labels) + '#' + '|'.join(self.hidden) +
                           f'#{len(self.Xr)}').encode()).hexdigest()[:10]
        slug = re.sub(r'[^a-z0-9]+', '_', '_'.join(self.hidden).lower())[:40]
        return Path(self.cache_dir) / f'base_{slug}_{key}.json'

    def train_base(self):
        p = self._cache_path()
        if p is not None and p.exists():
            b = xgb.Booster(); b.load_model(str(p)); b.set_param({'nthread': self.nthread}); return b, True
        b = train_booster(self.Xr, self.yr, len(self.labels), nthread=self.nthread)
        if p is not None: p.parent.mkdir(parents=True, exist_ok=True); b.save_model(str(p))
        return b, False

    def fit_novelty(self, booster):
        margins = booster.inplace_predict(self.Xv, predict_type='margin')
        self.novelty = NoveltyDetector(target_fpr=self.target_fpr, method=self.method).fit(self.Xr, self.Xv, margins, self.known)
        return self.novelty

    # ------------------------------------------------------------------ scoring
    def score(self, booster):
        P = booster.inplace_predict(self.Xs); M = booster.inplace_predict(self.Xs, predict_type='margin')
        kn = self.known + self.learned; Pk = P[:, kn] / P[:, kn].sum(1, keepdims=True)
        pred = np.array(kn)[Pk.argmax(1)]; p_att = 1 - Pk[:, kn.index(self.benign)]
        nov, unk, _ = self.novelty.score(self.Xs, M)
        if self.smoothing:                       # sustained novelty only: m of the last n windows (isolated hits = noise)
            m, n = self.smoothing; unk = unk & (np.convolve(unk.astype(int), np.ones(n, int))[:len(unk)] >= m)
        status = np.where(unk, 'Unknown pattern', np.where(p_att >= self.thr, 'Known-attack alert', 'Passed'))
        return pd.DataFrame({'true': self.true, 'pred': [self.labels[i] for i in pred], 'p_attack': p_att,
                             'novelty': nov, 'status': status, 'discovery': self.disc,
                             'hidden': np.isin(self.true, self.hidden)})

    def summary(self, S, half=None):
        if half is not None: S = S[S['discovery'] == (half == 'discovery')]
        h, b = S[S['hidden']], S[S['true'] == 'Benign']; ka = S[~S['hidden'] & (S['true'] != 'Benign')]
        pct = lambda m: float(m.mean() * 100) if len(m) else None
        return {'unseen_windows': int(len(h)),
                'unseen_flagged_unknown_%': pct(h['status'] == 'Unknown pattern'),
                'unseen_alerted_as_known_%': pct(h['status'] == 'Known-attack alert'),
                'unseen_caught_%': pct(h['status'] != 'Passed'),
                'unseen_missed_%': pct(h['status'] == 'Passed'),
                'benign_false_unknown_%': pct(b['status'] == 'Unknown pattern'),
                'benign_false_alert_%': pct(b['status'] == 'Known-attack alert'),
                'known_attacks_caught_%': pct(ka['status'] != 'Passed')}

    # ------------------------------------------------------------------ discovery + learning
    def discover(self, S, include_alerts=False, k='auto'):
        """Over-clusters the discovery-half UNKNOWN windows (k ~ n/30, 6..20): small pure patterns beat big mixed ones,
        and an analyst can give several patterns the same name. k=None -> silhouette choice."""
        m = S['discovery'].to_numpy() & ((S['status'] == 'Unknown pattern') | (include_alerts & (S['status'] == 'Known-attack alert'))).to_numpy()
        idx = np.flatnonzero(m)
        if len(idx) < 10: return idx, np.full(len(idx), -1), pd.DataFrame(), {}
        if k == 'auto': k = int(np.clip(len(idx) // 30, 6, 20))
        lab, summ, sil = discover_patterns(self.Xs[idx], self.novelty.space, self.Xr, [self.labels[i] for i in self.yr],
                                           self.features, k=k, seed=self.seed)
        if len(summ):                                                    # ground truth, for evaluation only
            summ['true_composition'] = [', '.join(f'{k} {v:.0f}%' for k, v in
                                                  (pd.Series(self.true[idx][lab == p]).value_counts(normalize=True) * 100).head(3).items())
                                        for p in summ['id']]
        return idx, lab, summ, sil

    def learn(self, booster, window_idx, name, mode='replay'):
        X_new = self.Xs[window_idx]
        bst, info = self.learner.learn(booster, X_new, name, mode=mode)
        if self.learner.labels != self.labels:                           # brand-new name appended a slot
            self.labels = list(self.learner.labels); self.L2I = {l: i for i, l in enumerate(self.labels)}
        c = self.L2I[name]
        if c not in self.known + self.learned: self.learned.append(c)
        self.novelty.add_reference(X_new, self.known + self.learned)
        self.novelty.recalibrate(self.Xv, bst.inplace_predict(self.Xv, predict_type='margin'))
        return bst, info

    def evaluate(self, booster):
        """Classification on the EVALUATION half (never labelled): old classes vs hidden / learned classes."""
        m = (~self.disc) & (self.ys >= 0)
        new = sorted({self.L2I[h] for h in self.hidden} | set(self.learned))
        out = evaluate(booster, self.Xs[m], self.ys[m], self.labels, known=self.known, new=new)
        hm = m & np.isin(self.true, self.hidden)                          # unseen windows now given a LEARNED label
        p = booster.inplace_predict(self.Xs[hm]).argmax(1) if hm.any() else np.array([])
        out['unseen_recognized_%'] = float(np.isin(p, self.learned).mean() * 100) if len(p) else None
        return out
