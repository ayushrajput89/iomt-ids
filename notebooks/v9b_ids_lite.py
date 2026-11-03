# =====================================================================
# CICIoMT2024 - v9b: IDS-Lite — accuracy vs latency / model-size trade-off
#   - Artifact guard + context buffer + ensemble (per-window XGB + context XGB)
#   - Streaming detector (one window at a time) + alert manager (m-of-n smoothing)
#   - System metrics: benign false alarms, attack detection rate, detection delay,
#     latency (ms/window, CPU, 1 thread), throughput, model size, RAM, per-alert XAI
# The v9 Full model (depth 8, <=1000 trees) was 209 MB and ~22 ms/window. Here we train smaller variants
# and check how much accuracy they lose and how much faster/smaller they are. GPU ~15-20 min, CPU ~40-60 min.
# =====================================================================

# %% CELL 1: Setup
import sys, os, time, re, glob, json, platform, resource, subprocess, warnings
from collections import deque
warnings.filterwarnings('ignore')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost', 'pyarrow'])
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT = '/content/drive/MyDrive/CICIoMT2024'
except ImportError:
    DATA_ROOT = os.environ.get('DATA_ROOT', r'C:\Users\YOUR_NAME\Documents\CICIoMT2024')
CACHE_DIR = os.path.join(DATA_ROOT, 'cache')
OUT_DIR   = os.path.join(DATA_ROOT, 'results_v9'); os.makedirs(OUT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(OUT_DIR, 'models');      os.makedirs(MODEL_DIR, exist_ok=True)
K, SEED = 10, 42
L_TEST  = [5, 10, 25, 50, 100, 500]
L_TRAIN = [5, 10, 25, 50, 100, 500, 1000]
try:
    HAS_GPU = subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
except FileNotFoundError:
    HAS_GPU = False
print('GPU:', HAS_GPU, '| CPUs:', os.cpu_count(), '| Data:', DATA_ROOT)

# %% CELL 2: Data (v5/v8 cache) + streams with segment info
train_b = pd.read_parquet(os.path.join(CACHE_DIR, 'train_blocks.parquet'))
test_b  = pd.read_parquet(os.path.join(CACHE_DIR, 'test_blocks.parquet'))
FEATS  = [c for c in train_b.columns if c not in ('label', 'block')]      # 39 shortcut-free (artifact guard)
LABELS = sorted(train_b['label'].unique()); L2I = {l: i for i, l in enumerate(LABELS)}
BENIGN = L2I['Benign']

def blocks_dict(df):
    d = {}
    for (lab, _), g in df.groupby(['label', 'block'], sort=False):
        d.setdefault(lab, []).append(g[FEATS].values.astype('float32'))
    return d

def add_ctx(X, k=K):
    roll = pd.DataFrame(X).rolling(k, min_periods=1)
    return np.hstack([X, roll.mean().values, roll.std().fillna(0).values]).astype('float32')

def make_stream(bd, L_choices, n_rows, seed, benign_p=0.5):
    """Mixed traffic. Returns X, y, segment id, position inside segment."""
    r = np.random.default_rng(seed)
    attacks = [l for l in bd if l != 'Benign']
    Xs, ys, seg, pos, total, sid = [], [], [], [], 0, 0
    first = list(r.permutation(list(bd)))
    while total < n_rows or first:
        lab = first.pop() if first else ('Benign' if r.random() < benign_p else attacks[r.integers(len(attacks))])
        blk = bd[lab][r.integers(len(bd[lab]))]
        L = min(int(r.choice(L_choices)), len(blk)); s = int(r.integers(0, len(blk) - L + 1))
        Xs.append(blk[s:s + L]); ys.append(np.full(L, L2I[lab])); seg.append(np.full(L, sid)); pos.append(np.arange(L))
        total += L; sid += 1
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(seg), np.concatenate(pos)

r = np.random.default_rng(SEED); bd_base, bd_meta = {}, {}
for lab, blks in blocks_dict(train_b).items():
    idx = r.permutation(len(blks)); cut = max(1, int(round(0.7 * len(blks))))
    bd_base[lab] = [blks[i] for i in idx[:cut]]
    bd_meta[lab] = [blks[i] for i in idx[cut:]] or [blks[i] for i in idx[:cut]]
bd_te = blocks_dict(test_b)
TEST = {L: make_stream(bd_te, [L], 200_000, seed=100 + L) for L in L_TEST}
print('Features:', len(FEATS), '| classes:', len(LABELS), '| test streams:', {L: len(v[1]) for L, v in TEST.items()})

# %% CELL 3: Helpers - train variant, evaluate, measure cost
from collections import deque

def fit_xgb(X, y, depth, n_est, lr):
    a, b = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=SEED)
    m = xgb.XGBClassifier(n_estimators=n_est, learning_rate=lr, max_depth=depth, subsample=0.8,
                          colsample_bytree=0.8, tree_method='hist', device='cuda' if HAS_GPU else 'cpu',
                          early_stopping_rounds=30, random_state=SEED, n_jobs=-1)
    m.fit(X[a], y[a], sample_weight=compute_sample_weight('balanced', y[a]), eval_set=[(X[b], y[b])], verbose=False)
    return m.get_booster()

def to_cpu(b, nthread=1):
    b.set_param({'device': 'cpu', 'nthread': nthread}); return b

def stream_latency(BA, BC, w, X, n=2000):
    buf = deque(maxlen=K); lat = []
    for i in range(n + 100):
        t1 = time.perf_counter()
        x = X[i]; buf.append(x); B = np.asarray(buf)
        std = B.std(0, ddof=1) if len(B) > 1 else np.zeros_like(x)
        p = w * BA.inplace_predict(x[None, :]) + (1 - w) * BC.inplace_predict(np.hstack([x, B.mean(0), std])[None, :].astype('float32'))
        int(p.argmax())
        if i >= 100: lat.append((time.perf_counter() - t1) * 1000)
    return np.array(lat)

def alerts_3of5(y, seg, pred):
    on = pd.Series((pred != BENIGN).astype(int)).rolling(5, min_periods=1).sum().values >= 3
    rise = on & ~np.r_[False, on[:-1]]; ben = y == BENIGN
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]; ends = np.r_[starts[1:], len(seg)]
    det, delay = [], []
    for s, e in zip(starts, ends):
        if y[s] == BENIGN: continue
        cnt = pd.Series((pred[s:e] != BENIGN).astype(int)).rolling(5, min_periods=1).sum().values
        hit = np.flatnonzero(cnt >= 3); det.append(len(hit) > 0)
        if len(hit): delay.append(hit[0])
    return round(rise[ben].sum() / ben.sum() * 1000, 2), round(np.mean(det) * 100, 2), round(float(np.mean(delay)), 2)

VARIANTS = [  # name, max_depth, max_trees, learning_rate
    ('Full (v9)', 8, 1000, 0.10),
    ('Lite-A',    6,  300, 0.15),
    ('Lite-B',    6,  150, 0.20),
    ('Lite-C',    4,  200, 0.25),
    ('Tiny',      4,   80, 0.30),
]

# %% CELL 4: Train + evaluate all variants (the Full v9 models are loaded, not retrained)
Xtr, ytr, _, _ = make_stream(bd_base, L_TRAIN, 600_000, seed=SEED + 1)
Xva, yva, _, _ = make_stream(bd_meta, L_TEST, 150_000, seed=SEED + 1000)
Ctr, Cva = add_ctx(Xtr), add_ctx(Xva)
rows = []
for name, depth, n_est, lr in VARIANTS:
    t0 = time.time()
    tag = re.sub(r'\W+', '_', name).strip('_').lower()
    vdir = os.path.join(OUT_DIR, 'models' if name.startswith('Full') else f'models_{tag}')
    os.makedirs(vdir, exist_ok=True)
    pa_path, pc_path, w_path = [os.path.join(vdir, f) for f in ('m_window.json', 'm_context.json', 'fusion.json')]
    if all(os.path.exists(p) for p in (pa_path, pc_path, w_path)):
        BA, BC = xgb.Booster(), xgb.Booster(); BA.load_model(pa_path); BC.load_model(pc_path)
        W = json.load(open(w_path))['w']; train_s = json.load(open(w_path)).get('train_sec')
    else:
        BA = fit_xgb(Xtr, ytr, depth, n_est, lr); BC = fit_xgb(Ctr, ytr, depth, n_est, lr)
        to_cpu(BA, os.cpu_count()); to_cpu(BC, os.cpu_count())
        pa, pc = BA.inplace_predict(Xva), BC.inplace_predict(Cva)
        ws = np.round(np.arange(0, 1.01, 0.1), 1)
        W = float(ws[int(np.argmax([f1_score(yva, (w * pa + (1 - w) * pc).argmax(1), average='macro') for w in ws]))])
        train_s = round(time.time() - t0)
        BA.save_model(pa_path); BC.save_model(pc_path)
        json.dump({'w': W, 'K': K, 'features': FEATS, 'labels': LABELS, 'train_sec': train_s,
                   'variant': name, 'max_depth': depth, 'max_trees': n_est}, open(w_path, 'w'))
    to_cpu(BA, os.cpu_count()); to_cpu(BC, os.cpu_count())
    f1s = {}
    for L, (X, y, seg, pos) in TEST.items():
        pred = (W * BA.inplace_predict(X) + (1 - W) * BC.inplace_predict(add_ctx(X))).argmax(1)
        f1s[L] = f1_score(y, pred, average='macro')
        if L == 100: fa, det, dly = alerts_3of5(y, seg, pred)
    to_cpu(BA, 1); to_cpu(BC, 1)
    lat = stream_latency(BA, BC, W, TEST[100][0])
    ubj = 0
    for b, f in ((BA, 'w.ubj'), (BC, 'c.ubj')):
        p = os.path.join(vdir, f); b.save_model(p); ubj += os.path.getsize(p)
    rows.append({'variant': name, 'depth': depth, 'trees_window': BA.num_boosted_rounds(),
                 'trees_context': BC.num_boosted_rounds(), 'w': W,
                 'macro_f1_mean_L': round(float(np.mean(list(f1s.values()))), 4),
                 'macro_f1_L5': round(f1s[5], 4), 'macro_f1_L100': round(f1s[100], 4),
                 'FA_per_1000_benign_3of5_L100': fa, 'detected_%_3of5_L100': det, 'delay_mean_3of5_L100': dly,
                 'latency_ms_median_1thread': round(float(np.median(lat)), 2),
                 'latency_ms_p95_1thread': round(float(np.percentile(lat, 95)), 2),
                 'windows_per_s_1thread': round(1000 / float(lat.mean()), 1),
                 'size_MB_json': round((os.path.getsize(pa_path) + os.path.getsize(pc_path)) / 1e6, 2),
                 'size_MB_ubj': round(ubj / 1e6, 2), 'train_s': train_s})
    print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'v9b_lite_tradeoff.csv'), index=False)

# %% CELL 5: Trade-off table
T = pd.DataFrame(rows)
full = T.iloc[0]
T['delta_macro_f1_pts'] = ((T['macro_f1_mean_L'] - full['macro_f1_mean_L']) * 100).round(2)
T['speedup_x'] = (full['latency_ms_median_1thread'] / T['latency_ms_median_1thread']).round(1)
T['size_reduction_x'] = (full['size_MB_json'] / T['size_MB_json']).round(1)
T.to_csv(os.path.join(OUT_DIR, 'v9b_lite_tradeoff.csv'), index=False)
print(T[['variant', 'depth', 'trees_window', 'trees_context', 'macro_f1_mean_L', 'delta_macro_f1_pts',
         'macro_f1_L5', 'macro_f1_L100', 'FA_per_1000_benign_3of5_L100', 'detected_%_3of5_L100',
         'latency_ms_median_1thread', 'windows_per_s_1thread', 'speedup_x', 'size_MB_json', 'size_MB_ubj',
         'size_reduction_x']].to_string(index=False))
print('\nLite models saved in:', OUT_DIR, '(folders models_lite_a, models_lite_b, models_lite_c, models_tiny)')
