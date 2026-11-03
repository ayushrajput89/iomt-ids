# =====================================================================
# CICIoMT2024 - v9: Context-Aware IDS as a SYSTEM
#   - Artifact guard + context buffer + ensemble (per-window XGB + context XGB)
#   - Streaming detector (one window at a time) + alert manager (m-of-n smoothing)
#   - System metrics: benign false alarms, attack detection rate, detection delay,
#     latency (ms/window, CPU, 1 thread), throughput, model size, RAM, per-alert XAI
# Colab: training ~10 min on GPU, ~45-60 min on CPU only (run ONCE; models are saved to Drive).
# Agli baar models load ho jayenge, training skip.
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

# %% CELL 3: Train (or load) the IDS models
def fit_xgb(X, y):
    a, b = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=SEED)
    m = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=8, subsample=0.8,
                          colsample_bytree=0.8, tree_method='hist', device='cuda' if HAS_GPU else 'cpu',
                          early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
    m.fit(X[a], y[a], sample_weight=compute_sample_weight('balanced', y[a]), eval_set=[(X[b], y[b])], verbose=False)
    return m

pa_path, pc_path, w_path = [os.path.join(MODEL_DIR, f) for f in ('m_window.json', 'm_context.json', 'fusion.json')]
t0 = time.time()
if all(os.path.exists(p) for p in (pa_path, pc_path, w_path)):
    m_a, m_c = xgb.XGBClassifier(), xgb.XGBClassifier()
    m_a.load_model(pa_path); m_c.load_model(pc_path)
    W = json.load(open(w_path))['w']; TRAIN_SEC = json.load(open(w_path)).get('train_sec')
    print('Loaded saved models | w =', W)
else:
    Xtr, ytr, _, _ = make_stream(bd_base, L_TRAIN, 600_000, seed=SEED + 1)
    Xva, yva, _, _ = make_stream(bd_meta, L_TEST, 150_000, seed=SEED + 1000)
    m_a = fit_xgb(Xtr, ytr); print('per-window model done', f'{time.time()-t0:.0f}s')
    m_c = fit_xgb(add_ctx(Xtr), ytr); print('context model done', f'{time.time()-t0:.0f}s')
    pa, pc = m_a.predict_proba(Xva), m_c.predict_proba(add_ctx(Xva))
    ws = np.round(np.arange(0, 1.01, 0.1), 1)
    W = float(ws[int(np.argmax([f1_score(yva, (w * pa + (1 - w) * pc).argmax(1), average='macro') for w in ws]))])
    TRAIN_SEC = round(time.time() - t0)
    m_a.save_model(pa_path); m_c.save_model(pc_path)
    json.dump({'w': W, 'K': K, 'features': FEATS, 'labels': LABELS, 'train_sec': TRAIN_SEC,
               'trained_on_gpu': HAS_GPU}, open(w_path, 'w'))
    del Xtr, Xva
    print('Trained + saved | w =', W, f'| {TRAIN_SEC}s')

# inference hamesha CPU, 1 thread (edge-like)
BA, BC = m_a.get_booster(), m_c.get_booster()
for b in (BA, BC):
    b.set_param({'device': 'cpu', 'nthread': 1})

# %% CELL 4: Streaming IDS class (one window at a time) + correctness check
class StreamingIDS:
    """Causal detector: per window -> (class index, confidence). Context = last K windows incl. current."""
    def __init__(self, ba, bc, w, k=K):
        self.ba, self.bc, self.w, self.buf = ba, bc, w, deque(maxlen=k)
    def step(self, x):
        self.buf.append(x)
        B = np.asarray(self.buf)
        std = B.std(0, ddof=1) if len(B) > 1 else np.zeros_like(x)
        xa = x[None, :]
        xc = np.hstack([x, B.mean(0), std])[None, :].astype('float32')
        p = self.w * self.ba.inplace_predict(xa) + (1 - self.w) * self.bc.inplace_predict(xc)
        c = int(p.argmax()); return c, float(p[0, c])

def batch_predict(X):
    """Same as StreamingIDS over a whole stream (vectorised, used for evaluation)."""
    p = W * BA.inplace_predict(X) + (1 - W) * BC.inplace_predict(add_ctx(X))
    return p.argmax(1), p.max(1)

X100 = TEST[100][0]
ids = StreamingIDS(BA, BC, W)
stream_pred = np.array([ids.step(X100[i])[0] for i in range(2000)])
batch_pred = batch_predict(X100[:2000])[0]
agree = (stream_pred == batch_pred).mean()
print(f'Streaming vs batch agreement on 2000 windows: {agree*100:.2f}%')
assert agree > 0.99, 'Streaming and batch predictions do not match'

# %% CELL 5: Alert manager + system metrics on mixed streams
def smooth(pred, m, n):
    att = (pred != BENIGN).astype(int)
    return (pd.Series(att).rolling(n, min_periods=1).sum().values >= m)

def alert_metrics(y, seg, pred, m, n):
    on = smooth(pred, m, n)
    rise = on & ~np.r_[False, on[:-1]]                      # new alarm event
    ben = y == BENIGN
    out = {'benign_windows_in_alarm_%': round(on[ben].mean() * 100, 2),
           'false_alarms_per_1000_benign': round(rise[ben].sum() / ben.sum() * 1000, 2)}
    det, delay, typed = [], [], []
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]
    ends = np.r_[starts[1:], len(seg)]
    for s, e in zip(starts, ends):
        if y[s] == BENIGN: continue
        # count only the windows of this attack segment (an alarm from the previous segment is not inherited)
        ps = pred[s:e]
        cnt = pd.Series((ps != BENIGN).astype(int)).rolling(n, min_periods=1).sum().values
        hit = np.flatnonzero(cnt >= m)
        det.append(len(hit) > 0)
        if len(hit):
            p0 = hit[0]; delay.append(p0)
            win = ps[max(0, p0 - n + 1):p0 + 1]; win = win[win != BENIGN]
            typed.append(np.bincount(win, minlength=len(LABELS)).argmax() == y[s])
    out.update({'attack_segments': len(det),
                'attack_segments_detected_%': round(np.mean(det) * 100, 2),
                'correct_type_at_first_alert_%': round(np.mean(typed) * 100, 2) if typed else None,
                'detection_delay_median_windows': float(np.median(delay)) if delay else None,
                'detection_delay_mean_windows': round(float(np.mean(delay)), 2) if delay else None})
    return out

SMOOTH = [(1, 1), (2, 3), (3, 5), (5, 10)]
rows, PRED = [], {}
for L, (X, y, seg, pos) in TEST.items():
    pred, conf = batch_predict(X); PRED[L] = pred
    base = {'L': L, 'macro_f1': round(f1_score(y, pred, average='macro'), 4),
            'accuracy': round(accuracy_score(y, pred), 4)}
    for m, n in SMOOTH:
        rows.append({**base, 'smoothing': f'{m}-of-{n}', **alert_metrics(y, seg, pred, m, n)})
    print(f'L={L} done')
SYS = pd.DataFrame(rows); SYS.to_csv(os.path.join(OUT_DIR, 'v9_system_metrics.csv'), index=False)
print(SYS.to_string(index=False))

# %% CELL 6: Cost - latency (1 CPU thread), throughput, model size, RAM
ids = StreamingIDS(BA, BC, W)
for i in range(200): ids.step(X100[i])                       # warm-up
lat = []
for i in range(200, 5200):
    t1 = time.perf_counter(); ids.step(X100[i]); lat.append((time.perf_counter() - t1) * 1000)
lat = np.array(lat)
for b in (BA, BC): b.set_param({'nthread': os.cpu_count()})
t1 = time.perf_counter(); batch_predict(X100); tb = time.perf_counter() - t1
for b in (BA, BC): b.set_param({'nthread': 1})
size_mb = sum(os.path.getsize(p) for p in (pa_path, pc_path)) / 1e6
COST = {'cpu': platform.processor() or platform.machine(), 'cpu_count': os.cpu_count(),
        'stream_latency_ms_median': round(float(np.median(lat)), 3),
        'stream_latency_ms_p95': round(float(np.percentile(lat, 95)), 3),
        'stream_latency_ms_p99': round(float(np.percentile(lat, 99)), 3),
        'stream_throughput_windows_per_s_1thread': round(1000 / float(np.mean(lat)), 1),
        'batch_throughput_windows_per_s_allcores': round(len(X100) / tb, 1),
        'model_size_MB_json': round(size_mb, 2),
        'trees_window_model': BA.num_boosted_rounds(), 'trees_context_model': BC.num_boosted_rounds(),
        'peak_RAM_MB_process': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'train_time_s': TRAIN_SEC, 'fusion_w': W, 'K': K}
json.dump(COST, open(os.path.join(OUT_DIR, 'v9_cost.json'), 'w'), indent=1)
print(json.dumps(COST, indent=1))

# %% CELL 7: Per-alert explanation (XAI) - top-3 SHAP reasons from XGBoost (no extra library)
def explain(x, cls, top=3):
    contrib = BA.predict(xgb.DMatrix(x[None, :]), pred_contribs=True)   # (1, C, F+1) for multi-class
    v = contrib[0, cls, :-1]
    idx = np.argsort(-np.abs(v))[:top]
    return ', '.join(f"{FEATS[i]} ({'+' if v[i] > 0 else '-'}{abs(v[i]):.2f})" for i in idx)

X, y, seg, pos = TEST[100]; pred = PRED[100]; on = smooth(pred, 3, 5)
rise = np.flatnonzero(on & ~np.r_[False, on[:-1]])
examples, seen = [], set()
for t in rise:
    if y[t] != BENIGN and pred[t] == y[t] and pred[t] not in seen:
        t1 = time.perf_counter(); why = explain(X[t], int(pred[t])); ms = (time.perf_counter() - t1) * 1000
        examples.append({'window': int(t), 'alert_type': LABELS[pred[t]], 'true': LABELS[y[t]],
                         'top3_reasons': why, 'explain_ms': round(ms, 2)})
        seen.add(pred[t])
    if len(examples) >= 6: break
EX = pd.DataFrame(examples); EX.to_csv(os.path.join(OUT_DIR, 'v9_alert_explanations.csv'), index=False)
print(EX.to_string(index=False))
print('\nSaare results:', OUT_DIR)

# %% CELL 8: Demo stream export (small sample for the VS Code project / dashboard)
X, y, seg, pos = TEST[100]
n_demo = 20_000
demo = pd.DataFrame(X[:n_demo], columns=FEATS)
demo['label'] = [LABELS[i] for i in y[:n_demo]]; demo['segment'] = seg[:n_demo]
demo.to_parquet(os.path.join(OUT_DIR, 'demo_stream.parquet'), index=False)
print('Demo stream saved:', demo.shape, '->', os.path.join(OUT_DIR, 'demo_stream.parquet'))
print('Download for the VS Code project: results_v9/models/ (3 files) + results_v9/demo_stream.parquet')
