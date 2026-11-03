# =====================================================================
# CICIoMT2024 - v10d
#   Part 1: Alert operating point at realistic base rate (95% benign)
#           - attack-score threshold sweep (precision vs detection)
#           - event-level alarm precision
#           - which attack class benign windows get confused with
#   Part 2: Why does IAT transfer to the official TEST set?
#           - per-class IAT ranges train vs test (overlap)
#           - IAT-only model trained on train, tested on official test
# Standalone. Uses saved v9 models (Lite-B always; Full too if GPU). CPU ~10-15 min, T4 ~10 min.
# Output: MyDrive/CICIoMT2024/results_v10d/
# =====================================================================

# ==== CELL 1: Setup ====
import sys, os, json, time, subprocess, warnings
warnings.filterwarnings('ignore')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost', 'pyarrow'])
import numpy as np, pandas as pd, xgboost as xgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT = '/content/drive/MyDrive/CICIoMT2024'
except ImportError:
    DATA_ROOT = os.environ.get('DATA_ROOT', r'E:\CICIoMT2024')
CACHE_DIR = os.path.join(DATA_ROOT, 'cache'); V9_DIR = os.path.join(DATA_ROOT, 'results_v9')
OUT_DIR = os.path.join(DATA_ROOT, 'results_v10d'); os.makedirs(OUT_DIR, exist_ok=True)
try:
    import torch; HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
SMOKE = os.environ.get('SMOKE') == '1'
K, M, N = 10, 3, 5
BLUE, ORANGE = '#2a78d6', '#eb6834'
print('GPU:', HAS_GPU, '| Data:', DATA_ROOT)

# ==== CELL 2: PART 1 - streams + model scores ====
te_b = pd.read_parquet(os.path.join(CACHE_DIR, 'test_blocks.parquet'))
FEATS = [c for c in te_b.columns if c not in ('label', 'block')]
LABELS = sorted(te_b['label'].unique()); L2I = {l: i for i, l in enumerate(LABELS)}; BENIGN = L2I['Benign']
bd_te = {}
for (lab, _), g in te_b.groupby(['label', 'block'], sort=False):
    bd_te.setdefault(lab, []).append(g[FEATS].values.astype('float32'))

def make_stream(bd, L_choices, n_rows, seed, benign_p=0.5):          # same generator as v9/v10c
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
    meta = json.load(open(os.path.join(vdir, 'fusion.json')))
    ba, bc = xgb.Booster(), xgb.Booster()
    ba.load_model(os.path.join(vdir, 'm_window.json')); bc.load_model(os.path.join(vdir, 'm_context.json'))
    for b in (ba, bc): b.set_param({'device': 'cuda' if HAS_GPU else 'cpu'})
    assert meta['features'] == FEATS and meta['labels'] == LABELS, 'feature/label order mismatch'
    return ba, bc, float(meta['w'])

VARIANTS = [('Lite-B', 'models_lite_b')] + ([('Full', 'models')] if HAS_GPU else [])
VARIANTS = [(n, os.path.join(V9_DIR, d)) for n, d in VARIANTS if os.path.exists(os.path.join(V9_DIR, d, 'fusion.json'))]
print('Variants:', [n for n, _ in VARIANTS], '(Full only on GPU - too slow on CPU)')
N_ROWS = 20_000 if SMOKE else 200_000
STREAMS = {(bp, L): make_stream(bd_te, [L], N_ROWS, seed=100 + L, benign_p=bp) for bp in (0.95, 0.5) for L in (25, 100)}
SCORES = {}
for vname, vdir in VARIANTS:
    ba, bc, w = load_variant(vdir); t0 = time.time()
    for key, (X, y, seg) in STREAMS.items():
        SCORES[(vname, *key)] = w * ba.inplace_predict(X) + (1 - w) * bc.inplace_predict(add_ctx(X))
    print(f'{vname} scored | {time.time()-t0:.0f}s')

# ==== CELL 3: PART 1 - threshold sweep, event precision, benign confusion ====
def runs(on):
    """start/end indices of contiguous True runs"""
    d = np.diff(np.r_[0, on.astype(int), 0]); return np.flatnonzero(d == 1), np.flatnonzero(d == -1)

def system_metrics(y, seg, flag):
    on = pd.Series(flag.astype(int)).rolling(N, min_periods=1).sum().values >= M
    ben = y == BENIGN; s_, e_ = runs(on)
    ev_true = np.array([(~ben[s:e]).any() for s, e in zip(s_, e_)], dtype=bool)
    rise_ben = sum(1 for s in s_ if ben[s])
    det, delay = [], []
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]; ends = np.r_[starts[1:], len(seg)]
    for s, e in zip(starts, ends):
        if y[s] == BENIGN: continue
        cnt = pd.Series(flag[s:e].astype(int)).rolling(N, min_periods=1).sum().values
        hit = np.flatnonzero(cnt >= M); det.append(len(hit) > 0)
        if len(hit): delay.append(hit[0])
    return {'false_alarms_per_1000_benign': rise_ben / ben.sum() * 1000,
            'benign_windows_in_alarm_%': on[ben].mean() * 100,
            'window_precision_%': (~ben[on]).mean() * 100 if on.any() else np.nan,
            'alarm_events': len(s_), 'event_precision_%': ev_true.mean() * 100 if len(s_) else np.nan,
            'detected_%': np.mean(det) * 100, 'delay_median': float(np.median(delay)) if delay else np.nan}

THR = ['argmax (current)', 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
rows = []
for (vname, bp, L), P in SCORES.items():
    X, y, seg = STREAMS[(bp, L)]; score = 1 - P[:, BENIGN]
    for t in THR:
        flag = (P.argmax(1) != BENIGN) if t == 'argmax (current)' else (score >= t)
        rows.append({'model': vname, 'benign_share': bp, 'L': L, 'threshold': str(t), **system_metrics(y, seg, flag)})
SW = pd.DataFrame(rows).round(2); SW.to_csv(os.path.join(OUT_DIR, 'P1_threshold_sweep.csv'), index=False)
print('PART 1a - threshold sweep on P(attack), 3-of-5 smoothing:')
for (vname, bp), g in SW.groupby(['model', 'benign_share'], sort=False):
    print(f'\n[{vname} | benign share {bp}] (mean over L=25,100)')
    print(g.groupby('threshold', sort=False)[['false_alarms_per_1000_benign', 'benign_windows_in_alarm_%', 'window_precision_%',
                                               'event_precision_%', 'detected_%', 'delay_median']].mean().round(2).to_string())

# Benign windows predicted as attack: which class?
bc_rows = []
for (vname, bp, L), P in SCORES.items():
    if bp != 0.95: continue
    y = STREAMS[(bp, L)][1]; pred = P.argmax(1); fp = (y == BENIGN) & (pred != BENIGN)
    vc = pd.Series(np.array(LABELS)[pred[fp]]).value_counts(normalize=True).mul(100)
    for lab, v in vc.items(): bc_rows.append({'model': vname, 'L': L, 'predicted_as': lab, 'share_of_benign_FPs_%': v})
BCF = pd.DataFrame(bc_rows).groupby(['model', 'predicted_as'])['share_of_benign_FPs_%'].mean().round(2) \
        .reset_index().sort_values(['model', 'share_of_benign_FPs_%'], ascending=[True, False])
BCF.to_csv(os.path.join(OUT_DIR, 'P1_benign_false_positive_classes.csv'), index=False)
print('\nPART 1b - benign windows wrongly flagged: predicted class (% of benign false positives, 95% benign):')
print(BCF.groupby('model').head(6).to_string(index=False))

# Figure: precision vs detection at 95% benign (Lite-B, L=100)
g = SW[(SW['model'] == VARIANTS[0][0]) & (SW['benign_share'] == 0.95) & (SW['L'] == 100)]
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(g['detected_%'], g['window_precision_%'], marker='o', color=BLUE, lw=2, label='window precision')
ax.plot(g['detected_%'], g['event_precision_%'], marker='s', color=ORANGE, lw=2, label='event precision')
for _, r in g.iterrows(): ax.annotate(r['threshold'].replace('argmax (current)', 'argmax'), (r['detected_%'], r['window_precision_%']),
                                      textcoords='offset points', xytext=(4, -10), fontsize=7)
ax.set_xlabel('Attack segments detected (%)'); ax.set_ylabel('Alarm precision (%)')
ax.set_title(f'{VARIANTS[0][0]}, 95% benign traffic, L=100', fontsize=10)
ax.grid(alpha=0.25); ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'fig_operating_point_95benign.png'), dpi=300); plt.close()

# ==== CELL 4: PART 2 - why IAT transfers to the official test set ====
train_s = pd.read_parquet(os.path.join(CACHE_DIR, 'train_sample50000.parquet'))
test_s  = pd.read_parquet(os.path.join(CACHE_DIR, 'test_full.parquet'))
def q(s): return s.quantile(0.05), s.quantile(0.5), s.quantile(0.95)
rows = []
for lab in LABELS:
    a5, a50, a95 = q(train_s.loc[train_s['label'] == lab, 'IAT']); b5, b50, b95 = q(test_s.loc[test_s['label'] == lab, 'IAT'])
    inter = max(0.0, min(a95, b95) - max(a5, b5)); union = max(a95, b95) - min(a5, b5)
    rows.append({'label': lab, 'train_p5': a5, 'train_median': a50, 'train_p95': a95, 'test_p5': b5, 'test_median': b50,
                 'test_p95': b95, 'range_overlap_%': 100 * inter / union if union > 0 else 100.0})
IR = pd.DataFrame(rows); IR.to_csv(os.path.join(OUT_DIR, 'P2_iat_train_vs_test.csv'), index=False)
pd.set_option('display.float_format', lambda v: f'{v:,.4g}')
print('\nPART 2a - IAT per class, train vs official test (5th / 50th / 95th percentile):'); print(IR.to_string(index=False))

# how separable are classes by IAT alone, and does that carry over to test?
L2Is = {l: i for i, l in enumerate(LABELS)}
ytr, yte = train_s['label'].map(L2Is).values, test_s['label'].map(L2Is).values
res = []
for name, feats in [('IAT only', ['IAT']), ('Number + Weight', ['Number', 'Weight']), ('IAT + Number + Weight', ['IAT', 'Number', 'Weight'])]:
    m = xgb.XGBClassifier(n_estimators=20 if SMOKE else 300, max_depth=6, learning_rate=0.1, tree_method='hist',
                          device='cuda' if HAS_GPU else 'cpu', n_jobs=-1, random_state=42)
    m.fit(train_s[feats].values, ytr, sample_weight=compute_sample_weight('balanced', ytr))
    p = m.predict(test_s[feats].values)
    res.append({'features (trained on train, tested on official test)': name, 'accuracy_%': accuracy_score(yte, p) * 100,
                'macro_f1_%': f1_score(yte, p, average='macro') * 100})
IT = pd.DataFrame(res).round(2); IT.to_csv(os.path.join(OUT_DIR, 'P2_artifact_only_transfer.csv'), index=False)
print('\nPART 2b - artifact-only models, train -> official test:'); print(IT.to_string(index=False))
print('\n==== v10d DONE ==== Outputs:', OUT_DIR)
print('Paste: PART 1a, 1b, 2a, 2b')
