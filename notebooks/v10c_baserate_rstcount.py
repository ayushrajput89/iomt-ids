# =====================================================================
# CICIoMT2024 - v10c: (A) rst_count check  (B) realistic benign ratio for the alert system
#   A: Is rst_count (39.5% of Benign SHAP) a recording artifact like IAT?
#      - within-file monotonic trend (Spearman vs row order) - IAT = positive control
#      - ablation: shortcut-free 39 vs 38 (without rst_count)
#   B: Saved v9 IDS models (Full + Lite-B) on streams with benign share 50/80/95%
#      -> false alarms per 1000 benign windows, alarm precision, detection, delay
# Standalone. No training for B (loads results_v9 models). T4 ~10 min, CPU ~30-40 min.
# Output: MyDrive/CICIoMT2024/results_v10c/
# =====================================================================

# ==== CELL 1: Setup ====
import sys, os, json, time, subprocess, warnings
warnings.filterwarnings('ignore')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost', 'pyarrow'])
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT = '/content/drive/MyDrive/CICIoMT2024'
except ImportError:
    DATA_ROOT = os.environ.get('DATA_ROOT', r'E:\CICIoMT2024')
CACHE_DIR = os.path.join(DATA_ROOT, 'cache'); V9_DIR = os.path.join(DATA_ROOT, 'results_v9')
OUT_DIR = os.path.join(DATA_ROOT, 'results_v10c'); os.makedirs(OUT_DIR, exist_ok=True)
try:
    import torch; HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
SMOKE = os.environ.get('SMOKE') == '1'
DEPTH, NTREES = (8, 1000) if HAS_GPU else (6, 300)
if SMOKE: NTREES = 15
REDUNDANT, SHORTCUT = ['Drate', 'Srate', 'LLC'], ['IAT', 'Number', 'Weight']
print('GPU:', HAS_GPU, '| Data:', DATA_ROOT)

# ==== CELL 2: PART A - rst_count check ====
test_s  = pd.read_parquet(os.path.join(CACHE_DIR, 'test_full.parquet'))      # rows kept in file order
train_s = pd.read_parquet(os.path.join(CACHE_DIR, 'train_sample50000.parquet'))
LABELS = sorted(train_s['label'].unique()); L2I = {l: i for i, l in enumerate(LABELS)}
FREE39 = [c for c in train_s.columns if c != 'label' and c not in REDUNDANT + SHORTCUT]
CHECK = ['IAT', 'rst_count', 'urg_count', 'ack_count', 'syn_count', 'fin_count', 'Rate', 'Header_Length', 'Tot_sum']
CHECK = [c for c in CHECK if c in test_s.columns]

# A1: monotonic trend with row order inside each class (a counter/clock-like feature -> |rho| near 1)
rows = []
for lab, g in test_s.groupby('label', sort=True):
    pos = np.arange(len(g))
    for c in CHECK:
        v = g[c].values
        rho = spearmanr(pos, v).correlation if np.std(v) > 0 else np.nan
        rows.append({'label': lab, 'feature': c, 'spearman_vs_row_order': rho})
TR = pd.DataFrame(rows).pivot(index='label', columns='feature', values='spearman_vs_row_order')[CHECK].round(3)
TR.to_csv(os.path.join(OUT_DIR, 'A1_trend_vs_row_order.csv'))
print('A1 - Spearman(feature, row order) inside each test class (|rho| ~ 1 = clock/counter-like):')
print(TR.to_string())
print('\nMean |rho| over classes:'); print(TR.abs().mean().round(3).to_string())

# A2: rst_count distribution per class (train sample vs test)
desc = lambda d: d.groupby('label')['rst_count'].agg(median='median', p90=lambda s: s.quantile(0.9), zero_pct=lambda s: (s == 0).mean() * 100)
D = pd.concat({'train': desc(train_s), 'test': desc(test_s)}, axis=1).round(2)
D.to_csv(os.path.join(OUT_DIR, 'A2_rst_count_distribution.csv'))
print('\nA2 - rst_count per class (median, 90th pct, % zeros):'); print(D.to_string())

# A3: ablation - shortcut-free 39 vs 38 (no rst_count), seed 42
def fit_xgb(X, y, seed=42):
    a, b = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=seed)
    m = xgb.XGBClassifier(n_estimators=NTREES, learning_rate=0.1, max_depth=DEPTH, subsample=0.8, colsample_bytree=0.8,
                          tree_method='hist', device='cuda' if HAS_GPU else 'cpu', early_stopping_rounds=50,
                          random_state=seed, n_jobs=-1)
    m.fit(X[a], y[a], sample_weight=compute_sample_weight('balanced', y[a]), eval_set=[(X[b], y[b])], verbose=False)
    return m
ytr, yte = train_s['label'].map(L2I).values, test_s['label'].map(L2I).values
ab = []
for name, feats in [('Shortcut-free 39', FREE39), ('39 minus rst_count (38)', [c for c in FREE39 if c != 'rst_count'])]:
    t0 = time.time(); p = fit_xgb(train_s[feats].values, ytr).predict(test_s[feats].values)
    f1c = f1_score(yte, p, average=None, labels=range(len(LABELS)))
    ab.append({'features': name, 'accuracy': accuracy_score(yte, p) * 100, 'macro_f1': f1_score(yte, p, average='macro') * 100,
               'Benign_F1': f1c[L2I['Benign']] * 100, 'ARP_F1': f1c[L2I['ARP_Spoofing']] * 100})
    print(f'{name} done {time.time()-t0:.0f}s')
AB = pd.DataFrame(ab).round(2); AB.to_csv(os.path.join(OUT_DIR, 'A3_rst_count_ablation.csv'), index=False)
print('\nA3 - ablation (official test, %):'); print(AB.to_string(index=False))

# ==== CELL 3: PART B - load v9 IDS models + build streams ====
K = 10; L_SYS = [25, 100]; BENIGN_P = [0.5, 0.8, 0.95]
te_b = pd.read_parquet(os.path.join(CACHE_DIR, "test_blocks.parquet"))          # test traffic only
FEATS = [c for c in te_b.columns if c not in ('label', 'block')]
BENIGN = L2I['Benign']
bd_te = {}
for (lab, _), g in te_b.groupby(['label', 'block'], sort=False):
    bd_te.setdefault(lab, []).append(g[FEATS].values.astype('float32'))

def make_stream(bd, L_choices, n_rows, seed, benign_p=0.5):          # same generator as v9
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
    assert meta['features'] == FEATS, 'feature order mismatch'
    return ba, bc, float(meta['w'])

VARIANTS = {n: os.path.join(V9_DIR, d) for n, d in [('Full', 'models'), ('Lite-B', 'models_lite_b')]
            if os.path.exists(os.path.join(V9_DIR, d, 'fusion.json'))}
print('Model variants found:', list(VARIANTS))
N_ROWS = 20_000 if SMOKE else 200_000

# ==== CELL 4: PART B - alert metrics (same definitions as v9) ====
def smooth(pred, m, n):
    return pd.Series((pred != BENIGN).astype(int)).rolling(n, min_periods=1).sum().values >= m

def alert_metrics(y, seg, pred, m, n):
    on = smooth(pred, m, n); rise = on & ~np.r_[False, on[:-1]]; ben = y == BENIGN
    out = {'false_alarms_per_1000_benign': rise[ben].sum() / ben.sum() * 1000,
           'benign_windows_in_alarm_%': on[ben].mean() * 100,
           'alarm_precision_%': (~ben[on]).mean() * 100 if on.any() else np.nan}   # share of alarm windows that are real attacks
    det, delay = [], []
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]; ends = np.r_[starts[1:], len(seg)]
    for s, e in zip(starts, ends):
        if y[s] == BENIGN: continue
        cnt = pd.Series((pred[s:e] != BENIGN).astype(int)).rolling(n, min_periods=1).sum().values
        hit = np.flatnonzero(cnt >= m); det.append(len(hit) > 0)
        if len(hit): delay.append(hit[0])
    out.update({'attack_segments': len(det), 'detected_%': np.mean(det) * 100,
                'delay_median': float(np.median(delay)) if delay else np.nan})
    return out

rows = []
for vname, vdir in VARIANTS.items():
    ba, bc, w = load_variant(vdir); t0 = time.time()
    for bp in BENIGN_P:
        for L in L_SYS:
            X, y, seg = make_stream(bd_te, [L], N_ROWS, seed=100 + L, benign_p=bp)
            p = w * ba.inplace_predict(X) + (1 - w) * bc.inplace_predict(add_ctx(X)); pred = p.argmax(1)
            base = {'model': vname, 'benign_share': bp, 'L': L, 'benign_windows_%': (y == BENIGN).mean() * 100,
                    'macro_f1': f1_score(y, pred, average='macro') * 100}
            for m, n in [(1, 1), (3, 5)]:
                rows.append({**base, 'smoothing': f'{m}-of-{n}', **alert_metrics(y, seg, pred, m, n)})
            print(f'{vname} benign_p={bp} L={L} done | {time.time()-t0:.0f}s')
SYSB = pd.DataFrame(rows).round(2); SYSB.to_csv(os.path.join(OUT_DIR, 'B_baserate_system.csv'), index=False)
cols = ['model', 'benign_share', 'L', 'smoothing', 'macro_f1', 'false_alarms_per_1000_benign', 'benign_windows_in_alarm_%',
        'alarm_precision_%', 'detected_%', 'delay_median']
print('\nPART B - alert system vs benign share of traffic:'); print(SYSB[cols].to_string(index=False))
print('\n==== v10c DONE ==== Outputs:', OUT_DIR)
print('Paste: A1 table + mean |rho|, A2, A3, PART B table')
