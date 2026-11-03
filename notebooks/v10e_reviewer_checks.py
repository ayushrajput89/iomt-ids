# =====================================================================
# CICIoMT2024 - v10e: "reviewer checks" (all CPU-friendly, GPU used if present)
#   P1 Single-feature scan: every feature alone, train -> official test
#   P2 Ablation: which artifact removal matters (IAT vs Number/Weight)
#   P3 Temporal baseline: rolling-average of per-window probabilities vs our context ensemble
#   P4 Alert threshold chosen on VALIDATION (+ separate ARP threshold), reported on test
#   P5 Leave-one-recording-out: does the model (and IAT) generalise to an unseen capture?
#   P6 Zero-day: leave-one-attack-family-out binary detection
#   P7 Calibration: Full vs Lite-B (ECE, Brier)
#   P8 Bootstrap 95% CIs for headline numbers
# Each part saves its CSV; re-running skips finished parts (safe after a disconnect).
# Needs: cache/{train_sample50000,test_full,train_blocks,test_blocks,fingerprint_train_files}.parquet,
#        results_v9/{models,models_lite_b}, results_v10a/probs_seed42.npz
# Output: MyDrive/CICIoMT2024/results_v10e/      CPU ~60-90 min | T4 ~25-35 min
# =====================================================================

# ==== CELL 1: Setup ====
import sys, os, json, time, subprocess, warnings
warnings.filterwarnings('ignore')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost', 'pyarrow'])
import numpy as np, pandas as pd, xgboost as xgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT = '/content/drive/MyDrive/CICIoMT2024'
except ImportError:
    DATA_ROOT = os.environ.get('DATA_ROOT', r'E:\CICIoMT2024')
CACHE_DIR = os.path.join(DATA_ROOT, 'cache'); V9_DIR = os.path.join(DATA_ROOT, 'results_v9')
OUT_DIR = os.path.join(DATA_ROOT, 'results_v10e'); os.makedirs(OUT_DIR, exist_ok=True)
try:
    import torch; HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
SMOKE = os.environ.get('SMOKE') == '1'
DEV = 'cuda' if HAS_GPU else 'cpu'
K, M, N = 10, 3, 5
REDUNDANT, SHORTCUT = ['Drate', 'Srate', 'LLC'], ['IAT', 'Number', 'Weight']
BLUE, ORANGE, RED = '#2a78d6', '#eb6834', '#d03b3b'
rng = np.random.default_rng(0)
print('GPU:', HAS_GPU, '| Data:', DATA_ROOT)

def out(name): return os.path.join(OUT_DIR, name)
def done(name): return os.path.exists(out(name))

def xgb_fit(X, y, n_est=300, depth=6, seed=42, early=True):
    kw = dict(n_estimators=(15 if SMOKE else n_est), learning_rate=0.1, max_depth=depth, subsample=0.8,
              colsample_bytree=0.8, tree_method='hist', device=DEV, random_state=seed, n_jobs=-1)
    if early:
        a, b = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=seed)
        m = xgb.XGBClassifier(early_stopping_rounds=30, **kw)
        m.fit(X[a], y[a], sample_weight=compute_sample_weight('balanced', y[a]), eval_set=[(X[b], y[b])], verbose=False)
    else:
        m = xgb.XGBClassifier(**kw); m.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    return m

def per_class_cap(df, cap, col='label', seed=0):
    df = df.sample(frac=1.0, random_state=seed); return df[df.groupby(col).cumcount() < cap]

def family6(l):
    if l == 'Benign': return 'Benign'
    if l.startswith('MQTT'): return 'MQTT'
    if l.startswith('Recon'): return 'Recon'
    if 'Spoof' in l: return 'Spoofing'
    return 'DDoS' if l.startswith('DDoS') else 'DoS'

# ==== CELL 2: Static data ====
train_s = pd.read_parquet(os.path.join(CACHE_DIR, 'train_sample50000.parquet'))
test_s  = pd.read_parquet(os.path.join(CACHE_DIR, 'test_full.parquet'))
LABELS = sorted(train_s['label'].unique()); L2I = {l: i for i, l in enumerate(LABELS)}; LAB = np.array(LABELS)
ALL45 = [c for c in train_s.columns if c != 'label']
ALL42 = [c for c in ALL45 if c not in REDUNDANT]
FREE39 = [c for c in ALL42 if c not in SHORTCUT]
CAP = 2000 if SMOKE else (50_000 if HAS_GPU else 20_000)          # rows per class for static models
tr = per_class_cap(train_s, CAP); ytr = tr['label'].map(L2I).values; yte = test_s['label'].map(L2I).values
print('static train rows', len(tr), '| test rows', len(test_s))

# ==== CELL 3: P1 single-feature scan ====
if not done('P1_single_feature_scan.csv'):
    t0 = time.time(); trs = per_class_cap(train_s, 500 if SMOKE else 10_000); ys = trs['label'].map(L2I).values
    rows = []
    for f in ALL45:
        m = xgb_fit(trs[[f]].values, ys, n_est=100, early=False); p = m.predict(test_s[[f]].values)
        rows.append({'feature': f, 'accuracy_%': accuracy_score(yte, p) * 100, 'macro_f1_%': f1_score(yte, p, average='macro') * 100})
    pd.DataFrame(rows).sort_values('accuracy_%', ascending=False).round(2).to_csv(out('P1_single_feature_scan.csv'), index=False)
    print(f'P1 done {time.time()-t0:.0f}s')
P1 = pd.read_csv(out('P1_single_feature_scan.csv'))
print('\nP1 - each feature ALONE (train -> official test, 19 classes), top 12:'); print(P1.head(12).to_string(index=False))
print(f"Median single-feature accuracy: {P1['accuracy_%'].median():.2f}% | IAT rank: {int(np.flatnonzero(P1['feature'].values == 'IAT')[0]) + 1}")
fig, ax = plt.subplots(figsize=(6.5, 7.5)); t = P1.iloc[::-1]
ax.barh(t['feature'], t['accuracy_%'], color=[ORANGE if f in SHORTCUT else BLUE for f in t['feature']], height=0.7)
ax.set_xlabel('19-class accuracy using ONE feature (%)'); ax.tick_params(axis='y', labelsize=7)
ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=0.25)
plt.tight_layout(); plt.savefig(out('fig_single_feature_scan.png'), dpi=300); plt.close()

# ==== CELL 4: P2 ablation ====
if not done('P2_ablation.csv'):
    rows = []
    for name, feats in [('All 42', ALL42), ('42 minus IAT (41)', [c for c in ALL42 if c != 'IAT']),
                        ('42 minus Number+Weight (40)', [c for c in ALL42 if c not in ('Number', 'Weight')]),
                        ('Shortcut-free 39 (minus all three)', FREE39)]:
        t0 = time.time(); p = xgb_fit(tr[feats].values, ytr).predict(test_s[feats].values)
        rows.append({'features': name, 'accuracy_%': accuracy_score(yte, p) * 100, 'macro_f1_%': f1_score(yte, p, average='macro') * 100})
        print(f'P2 {name} {time.time()-t0:.0f}s')
    pd.DataFrame(rows).round(2).to_csv(out('P2_ablation.csv'), index=False)
print('\nP2 - ablation (train -> official test):'); print(pd.read_csv(out('P2_ablation.csv')).to_string(index=False))

# ==== CELL 5: Stream data + saved IDS models ====
tr_b = pd.read_parquet(os.path.join(CACHE_DIR, 'train_blocks.parquet'))
te_b = pd.read_parquet(os.path.join(CACHE_DIR, 'test_blocks.parquet'))
FEATS = [c for c in te_b.columns if c not in ('label', 'block')]; BENIGN = L2I['Benign']; ARP = L2I['ARP_Spoofing']
def blocks_dict(df):
    d = {}
    for (lab, _), g in df.groupby(['label', 'block'], sort=False): d.setdefault(lab, []).append(g[FEATS].values.astype('float32'))
    return d
r = np.random.default_rng(42); bd_meta = {}
for lab, blks in blocks_dict(tr_b).items():                    # same 70/30 block split as v8/v9 -> meta = validation
    idx = r.permutation(len(blks)); cut = max(1, int(round(0.7 * len(blks))))
    bd_meta[lab] = [blks[i] for i in idx[cut:]] or [blks[i] for i in idx[:cut]]
bd_te = blocks_dict(te_b)
def make_stream(bd, L_choices, n_rows, seed, benign_p=0.5):
    r = np.random.default_rng(seed); attacks = [l for l in bd if l != 'Benign']
    Xs, ys, seg, total, sid = [], [], [], 0, 0; first = list(r.permutation(list(bd)))
    while total < n_rows or first:
        lab = first.pop() if first else ('Benign' if r.random() < benign_p else attacks[r.integers(len(attacks))])
        blk = bd[lab][r.integers(len(bd[lab]))]
        L = min(int(r.choice(L_choices)), len(blk)); s = int(r.integers(0, len(blk) - L + 1))
        Xs.append(blk[s:s + L]); ys.append(np.full(L, L2I[lab])); seg.append(np.full(L, sid)); total += L; sid += 1
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(seg)
def add_ctx(X, k=K):
    roll = pd.DataFrame(X).rolling(k, min_periods=1)
    return np.hstack([X, roll.mean().values, roll.std().fillna(0).values]).astype('float32')
def load_variant(vdir):
    meta = json.load(open(os.path.join(vdir, 'fusion.json'))); ba, bc = xgb.Booster(), xgb.Booster()
    ba.load_model(os.path.join(vdir, 'm_window.json')); bc.load_model(os.path.join(vdir, 'm_context.json'))
    for b in (ba, bc): b.set_param({'device': DEV})
    assert meta['features'] == FEATS and meta['labels'] == LABELS
    return ba, bc, float(meta['w'])
VARIANTS = [('Lite-B', 'models_lite_b')] + ([('Full', 'models')] if HAS_GPU else [])
VARIANTS = [(n, os.path.join(V9_DIR, d)) for n, d in VARIANTS if os.path.exists(os.path.join(V9_DIR, d, 'fusion.json'))]
NR = 20_000 if SMOKE else 200_000
print('\nStream variants:', [n for n, _ in VARIANTS])

# ==== CELL 6: P3 temporal baseline ====
def rolling_avg(P, k):
    return pd.DataFrame(P).rolling(k, min_periods=1).mean().values
if not done('P3_temporal_baseline.csv'):
    rows = []
    for vname, vdir in VARIANTS:
        ba, bc, w = load_variant(vdir)
        for L in [5, 10, 25, 50, 100, 500]:
            X, y, _ = make_stream(bd_te, [L], NR, seed=100 + L)          # same test streams as v8/v9
            pa = ba.inplace_predict(X); pe = w * pa + (1 - w) * bc.inplace_predict(add_ctx(X))
            for name, P in [('(a) per-window', pa), ('(a) + rolling-avg probs K=5', rolling_avg(pa, 5)),
                            ('(a) + rolling-avg probs K=10', rolling_avg(pa, 10)), ('(e) context ensemble (ours)', pe),
                            ('(e) + rolling-avg probs K=5', rolling_avg(pe, 5))]:
                rows.append({'model': vname, 'L': L, 'method': name, 'macro_f1_%': f1_score(y, P.argmax(1), average='macro') * 100})
        print(f'P3 {vname} done')
    pd.DataFrame(rows).round(2).to_csv(out('P3_temporal_baseline.csv'), index=False)
P3 = pd.read_csv(out('P3_temporal_baseline.csv'))
T3 = P3.pivot_table(index=['model', 'method'], columns='L', values='macro_f1_%', sort=False)
T3['mean over L'] = T3.mean(1); print('\nP3 - temporal baseline, mixed-stream macro-F1 (%):'); print(T3.round(2).to_string())

# ==== CELL 7: P4 threshold chosen on validation (+ ARP threshold) ====
def runs(on):
    d = np.diff(np.r_[0, on.astype(int), 0]); return np.flatnonzero(d == 1), np.flatnonzero(d == -1)
def sysm(y, seg, flag):
    on = pd.Series(flag.astype(int)).rolling(N, min_periods=1).sum().values >= M; ben = y == BENIGN; s_, e_ = runs(on)
    ev = np.array([(~ben[s:e]).any() for s, e in zip(s_, e_)], dtype=bool)
    det = []
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]; ends = np.r_[starts[1:], len(seg)]
    for s, e in zip(starts, ends):
        if y[s] != BENIGN: det.append((pd.Series(flag[s:e].astype(int)).rolling(N, min_periods=1).sum().values >= M).any())
    return {'FA_per_1000_benign': sum(1 for s in s_ if ben[s]) / ben.sum() * 1000, 'event_precision_%': ev.mean() * 100 if len(s_) else np.nan,
            'window_precision_%': (~ben[on]).mean() * 100 if on.any() else np.nan, 'detected_%': np.mean(det) * 100}
def flags(P, t, t_arp):
    score = 1 - P[:, BENIGN]; att = P.copy(); att[:, BENIGN] = -1; pc = att.argmax(1)
    return (score >= t) & ((pc != ARP) | (score >= t_arp))
GRID_T = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]; GRID_A = [None, 0.9, 0.95, 0.98, 0.99, 0.995]
if not done('P4_validated_threshold.csv'):
    rows = []
    for vname, vdir in VARIANTS:
        ba, bc, w = load_variant(vdir)
        def score(bd, seed, L, bp):
            X, y, seg = make_stream(bd, [L], NR, seed=seed, benign_p=bp)
            return w * ba.inplace_predict(X) + (1 - w) * bc.inplace_predict(add_ctx(X)), y, seg
        VAL = {L: score(bd_meta, 900 + L, L, 0.95) for L in (25, 100)}
        TST = {(bp, L): score(bd_te, 100 + L, L, bp) for bp in (0.95, 0.5) for L in (25, 100)}
        best = {}
        for mode, grid in [('global t', [(t, None) for t in GRID_T]),
                           ('global t + ARP t', [(t, a) for t in GRID_T for a in GRID_A if a is None or a > t])]:
            cand = []
            for t, a in grid:
                ms = [sysm(y, seg, flags(P, t, a if a else t)) for P, y, seg in VAL.values()]
                ep = np.mean([m['event_precision_%'] for m in ms]); dt = np.mean([m['detected_%'] for m in ms])
                cand.append((2 * ep * dt / (ep + dt) if ep + dt > 0 else 0, t, a))   # event-F1 on validation
            best[mode] = max(cand)[1:]
        for mode, (t, a) in [('argmax (no threshold)', (None, None))] + list(best.items()):
            for (bp, L), (P, y, seg) in TST.items():
                fl = (P.argmax(1) != BENIGN) if t is None else flags(P, t, a if a else t)
                rows.append({'model': vname, 'rule': mode, 't': t, 't_arp': a, 'benign_share': bp, 'L': L, **sysm(y, seg, fl)})
        print(f'P4 {vname} done | chosen on validation: {best}')
    pd.DataFrame(rows).round(3).to_csv(out('P4_validated_threshold.csv'), index=False)
P4 = pd.read_csv(out('P4_validated_threshold.csv'))
print('\nP4 - thresholds chosen on VALIDATION streams, reported on TEST (mean over L=25,100):')
print(P4.groupby(['model', 'benign_share', 'rule', 't', 't_arp'], dropna=False, sort=False)[
      ['FA_per_1000_benign', 'event_precision_%', 'window_precision_%', 'detected_%']].mean().round(2).to_string())

# ==== CELL 8: P5 leave-one-recording-out ====
if not done('P5_leave_one_recording_out.csv'):
    FP = pd.read_parquet(os.path.join(CACHE_DIR, 'fingerprint_train_files.parquet'))
    FP = FP.sample(frac=1.0, random_state=1); FP = FP[FP.groupby('file').cumcount() < (300 if SMOKE else 5000)]
    yfp = FP['label'].map(L2I).values
    multi = sorted(l for l, g in FP.groupby('label') if g['file'].nunique() >= 2)
    files_of = {l: sorted(FP.loc[FP['label'] == l, 'file'].unique()) for l in multi}
    rows = []
    for fold in range(2):
        held = [files_of[l][fold % len(files_of[l])] for l in multi]
        is_held = FP['file'].isin(held).values
        # in-session reference: same number of rows, random split from the SAME files
        ref_tr, ref_te = train_test_split(np.arange(len(FP)), test_size=is_held.sum() / len(FP), random_state=fold,
                                          stratify=FP['file'].values)
        for fname, feats in [('Shortcut-free 39', FREE39), ('IAT only', ['IAT'])]:
            for split, (a, b) in [('unseen recording (LORO)', (np.flatnonzero(~is_held), np.flatnonzero(is_held))),
                                  ('same recordings (random split)', (ref_tr, ref_te[np.isin(FP['label'].values[ref_te], multi)]))]:
                m = xgb_fit(FP[feats].values[a], yfp[a], n_est=200); p = m.predict(FP[feats].values[b]); yt = yfp[b]
                tm = lambda v: np.array([l.replace('DDoS', 'DoS') for l in LAB[v]])
                rows.append({'fold': fold, 'features': fname, 'split': split, 'acc_19_%': accuracy_score(yt, p) * 100,
                             'acc_13type_%': (tm(yt) == tm(p)).mean() * 100, 'n_test': len(b)})
        print(f'P5 fold {fold} done (held-out files: {len(held)})')
    pd.DataFrame(rows).round(2).to_csv(out('P5_leave_one_recording_out.csv'), index=False)
P5 = pd.read_csv(out('P5_leave_one_recording_out.csv'))
print('\nP5 - DoS/DDoS classes: test on an UNSEEN recording vs same recordings (mean of 2 folds):')
print(P5.groupby(['features', 'split'], sort=False)[['acc_19_%', 'acc_13type_%']].mean().round(2).to_string())

# ==== CELL 9: P6 zero-day (leave-one-attack-family-out, binary) ====
if not done('P6_zero_day.csv'):
    trz = per_class_cap(train_s, 500 if SMOKE else 10_000); fam_tr = trz['label'].map(family6).values; fam_te = test_s['label'].map(family6).values
    rows = []
    for held in ['DDoS', 'DoS', 'MQTT', 'Recon', 'Spoofing']:
        keep = fam_tr != held; yb = (fam_tr[keep] != 'Benign').astype(int)
        for fname, feats in [('All 42 (with artifacts)', ALL42), ('Shortcut-free 39', FREE39)]:
            m = xgb_fit(trz[feats].values[keep], yb, n_est=200); p = m.predict(test_s[feats].values)
            rows.append({'held_out_family': held, 'features': fname,
                         'recall_on_unseen_family_%': p[fam_te == held].mean() * 100,
                         'benign_FPR_%': p[fam_te == 'Benign'].mean() * 100})
        print(f'P6 {held} done')
    pd.DataFrame(rows).round(2).to_csv(out('P6_zero_day.csv'), index=False)
print('\nP6 - zero-day: binary detector trained WITHOUT one attack family:')
print(pd.read_csv(out('P6_zero_day.csv')).pivot(index='held_out_family', columns='features',
      values=['recall_on_unseen_family_%', 'benign_FPR_%']).round(2).to_string())

# ==== CELL 10: P7 calibration ====
def ece(conf, correct, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any(): e += s.mean() * abs(correct[s].mean() - conf[s].mean())
    return e * 100
if not done('P7_calibration.csv'):
    rows, REL = [], {}
    for vname, vdir in VARIANTS:
        ba, bc, w = load_variant(vdir); X, y, _ = make_stream(bd_te, [100], NR, seed=200, benign_p=0.95)
        P = w * ba.inplace_predict(X) + (1 - w) * bc.inplace_predict(add_ctx(X))
        att_true = (y != BENIGN).astype(float); att_p = 1 - P[:, BENIGN]
        rows.append({'model': vname, 'ECE_top_class_%': ece(P.max(1), (P.argmax(1) == y).astype(float)),
                     'ECE_attack_score_%': ece(np.maximum(att_p, 1 - att_p), ((att_p >= 0.5) == att_true).astype(float)),
                     'Brier_attack_score': np.mean((att_p - att_true) ** 2)})
        bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(att_p, bins) - 1, 0, 9)
        REL[vname] = [(att_p[idx == i].mean(), att_true[idx == i].mean()) for i in range(10) if (idx == i).sum() > 50]
    pd.DataFrame(rows).round(4).to_csv(out('P7_calibration.csv'), index=False)
    fig, ax = plt.subplots(figsize=(4.4, 4.2)); ax.plot([0, 1], [0, 1], ls='--', color='#9a9890', lw=1)
    for (vname, pts), c in zip(REL.items(), [BLUE, ORANGE]):
        ax.plot(*zip(*pts), marker='o', color=c, lw=2, label=vname)
    ax.set_xlabel('Predicted P(attack)'); ax.set_ylabel('Observed attack fraction'); ax.legend(frameon=False)
    ax.set_title('Reliability, 95% benign traffic', fontsize=10); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(out('fig_calibration.png'), dpi=300); plt.close()
print('\nP7 - calibration on 95%-benign test stream:'); print(pd.read_csv(out('P7_calibration.csv')).to_string(index=False))

# ==== CELL 11: P8 bootstrap 95% CIs ====
def boot(y, p, fn, B):
    n = len(y); vals = [fn(y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(vals, [2.5, 97.5])
if not done('P8_bootstrap_ci.csv'):
    B = 20 if SMOKE else 200; rows = []
    z = np.load(os.path.join(DATA_ROOT, 'results_v10a', 'probs_seed42.npz'), allow_pickle=True)
    ys, ps = z['y_static'], z['P_static'].argmax(1)
    sub = rng.choice(len(ys), min(len(ys), 400_000), replace=False)            # speed: CI on a 400k subsample
    tm = np.vectorize(lambda v: LABELS[v].replace('DDoS', 'DoS'))
    for name, fn, a, b in [('Static shortcut-free acc 19-class', accuracy_score, ys[sub], ps[sub]),
                           ('Static shortcut-free macro-F1 19-class', lambda u, v: f1_score(u, v, average='macro'), ys[sub], ps[sub]),
                           ('Static shortcut-free acc 13-type', lambda u, v: (tm(u) == tm(v)).mean(), ys[sub][:100_000], ps[sub][:100_000])]:
        lo, hi = boot(a, b, fn, B); rows.append({'metric': name, 'point': fn(a, b) * 100, 'ci_low': lo * 100, 'ci_high': hi * 100})
    for L in (25, 100):
        yv, pv = z[f'y_L{L}'], z[f'P_L{L}'].argmax(1)
        f = lambda u, v: f1_score(u, v, average='macro'); lo, hi = boot(yv, pv, f, B)
        rows.append({'metric': f'Stream ensemble macro-F1 L={L}', 'point': f(yv, pv) * 100, 'ci_low': lo * 100, 'ci_high': hi * 100})
    pd.DataFrame(rows).round(2).to_csv(out('P8_bootstrap_ci.csv'), index=False)
print('\nP8 - bootstrap 95% CIs (window resampling):'); print(pd.read_csv(out('P8_bootstrap_ci.csv')).to_string(index=False))
print('\n==== v10e DONE ==== Outputs:', OUT_DIR)
print('Paste everything printed from P1 to P8.')
