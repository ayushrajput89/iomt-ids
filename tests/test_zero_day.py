"""Phase-2 tests on small synthetic blobs (fast, no dataset needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from iomt_ids.continual import ContinualLearner, ModelRegistry, evaluate, train_booster   # noqa: E402
from iomt_ids.drift import drift_report, psi, summarize                                  # noqa: E402
from iomt_ids.novelty import NoveltyDetector                                             # noqa: E402
from iomt_ids.patterns import discover_patterns                                          # noqa: E402
from iomt_ids.zero_day import ZeroDayLab, split_discovery                                # noqa: E402

LABELS = ['ARP_Spoofing', 'Benign', 'DDoS-UDP', 'Recon-Port_Scan']
FEATS = [f'f{i}' for i in range(6)]
CENTERS = {'ARP_Spoofing': 8, 'Benign': 0, 'DDoS-UDP': 4, 'Recon-Port_Scan': -4}


def blobs(n, seed, labels=LABELS):
    r = np.random.default_rng(seed); rows = []
    for l in labels:
        X = r.normal(CENTERS[l], 0.6, (n, len(FEATS))); rows.append(pd.DataFrame(np.exp(X / 3), columns=FEATS).assign(label=l))
    return pd.concat(rows, ignore_index=True)


def test_novelty_flags_unseen_class():
    ref, val, unk = blobs(150, 0, LABELS[1:]), blobs(100, 1, LABELS[1:]), blobs(100, 2, ['ARP_Spoofing'])
    nd = NoveltyDetector(method='knn').fit(ref[FEATS].values, val[FEATS].values)
    _, flag_unk, _ = nd.score(unk[FEATS].values); _, flag_known, _ = nd.score(blobs(100, 3, LABELS[1:])[FEATS].values)
    assert flag_unk.mean() > 0.9 and flag_known.mean() < 0.15


def test_replay_learns_without_forgetting():
    L2I = {l: i for i, l in enumerate(LABELS)}; known = [1, 2, 3]
    tr = blobs(150, 0, LABELS[1:]); te = blobs(80, 5)
    base = train_booster(tr[FEATS].values, tr['label'].map(L2I).values, len(LABELS), rounds=40)
    Xn = blobs(60, 6, ['ARP_Spoofing'])[FEATS].values; yte = te['label'].map(L2I).values
    res = {}
    for mode in ('replay', 'no-replay'):
        cl = ContinualLearner(LABELS, tr[FEATS].values, tr['label'].map(L2I).values, replay_per_class=60)
        b, info = cl.learn(base, Xn, 'ARP_Spoofing', mode=mode, rounds=40)
        res[mode] = evaluate(b, te[FEATS].values, yte, LABELS, known=known, new=[0])
    assert res['replay']['new_macro_f1'] > 90 and res['replay']['old_macro_f1'] > 90
    assert res['no-replay']['old_macro_f1'] < res['replay']['old_macro_f1'] - 30      # catastrophic forgetting


def test_new_name_forces_retrain(tmp_path):
    L2I = {l: i for i, l in enumerate(LABELS)}; tr = blobs(80, 0)
    base = train_booster(tr[FEATS].values, tr['label'].map(L2I).values, len(LABELS), rounds=20)
    cl = ContinualLearner(LABELS, tr[FEATS].values, tr['label'].map(L2I).values)
    b, info = cl.learn(base, np.full((30, len(FEATS)), 50.0), 'New-Attack-1', mode='replay')
    assert info['mode'] == 'retrain' and len(cl.labels) == 5 and b.inplace_predict(tr[FEATS].values[:2]).shape[1] == 5
    reg = ModelRegistry(tmp_path); v = reg.save(b, cl.labels, info); assert v == 'v1' and len(reg.table()) == 1


def test_patterns_and_drift():
    r = np.random.default_rng(0); Xa = np.exp(r.normal(8, .5, (120, 6)) / 3); Xb = np.exp(r.normal(-8, .5, (120, 6)) / 3)
    ref = blobs(100, 0, LABELS[1:])
    nd = NoveltyDetector(method='knn').fit(ref[FEATS].values, ref[FEATS].values)
    lab, summ, _ = discover_patterns(np.vstack([Xa, Xb]), nd.space, ref[FEATS].values, ref['label'].values, FEATS)
    assert len(summ) == 2 and len(set(lab[:120])) == 1 and lab[0] != lab[-1]
    a = r.normal(0, 1, 5000); assert psi(a, r.normal(0, 1, 5000)) < 0.05 and psi(a, r.normal(2, 1, 5000)) > 0.25
    rep = drift_report(ref, ref.assign(f0=ref['f0'] * 50), FEATS); assert rep.iloc[0]['feature'] == 'f0'
    assert summarize(rep)['worst_feature'] == 'f0'


def test_lab_end_to_end():
    ref, val = blobs(120, 0), blobs(60, 1)
    st = blobs(100, 2)                                                   # contiguous 25-window segments per class
    st['segment'] = np.arange(len(st)) // 25
    order = np.random.default_rng(0).permutation(st['segment'].unique())  # shuffle whole segments, keep each contiguous
    st = pd.concat([st[st['segment'] == g] for g in order]).reset_index(drop=True)
    m = split_discovery(st); assert 0.3 < m.mean() < 0.7
    lab = ZeroDayLab(ref, val, st, FEATS, LABELS, ['ARP_Spoofing'], method='knn+energy')
    b, _ = lab.train_base(); lab.fit_novelty(b); S = lab.score(b); s0 = lab.summary(S)
    assert s0['unseen_caught_%'] > 80 and s0['benign_false_unknown_%'] < 20
    idx, plab, summ, _ = lab.discover(S, include_alerts=True); assert len(summ) >= 1
    p = summ[summ['true_composition'].str.startswith('ARP_Spoofing')].iloc[0]          # the analyst picks the new pattern
    b2, _ = lab.learn(b, idx[plab == p['id']], 'ARP_Spoofing', 'replay')
    assert lab.evaluate(b2)['new_macro_f1'] > lab.evaluate(b)['new_macro_f1']


def test_unknown_smoothing_suppresses_isolated_hits():
    ref, val = blobs(120, 0), blobs(60, 1)
    st = blobs(100, 2); st['segment'] = np.arange(len(st)) // 25
    lab = ZeroDayLab(ref, val, st, FEATS, LABELS, ['ARP_Spoofing'], unknown_smoothing=(3, 5))
    b, _ = lab.train_base(); lab.fit_novelty(b)
    lab.Xs = lab.Xs.copy(); lab.Xs[st.index[st['label'] == 'Benign'][::7]] *= 40               # isolated odd benign windows
    S = lab.score(b); ben = S['true'] == 'Benign'
    assert (S.loc[ben, 'status'] == 'Unknown pattern').mean() < 0.05
    assert (S.loc[S['hidden'], 'status'] == 'Unknown pattern').mean() > 0.8
