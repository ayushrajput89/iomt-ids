# =====================================================================
# CICIoMT2024 - v10a: Extension experiments (after CHECKPOINT 9 Oct)
#   Part 1: Label identifiability - metric hierarchy (binary / family / merged / 19)
#           + "how many errors are only DoS<->DDoS swaps of the same protocol"
#   Part 2: Recording-fingerprint test - identify the PCAP FILE from IAT/Number/Weight
#           (same-class files: same attack, only the recording differs) + distribution plots
#   Part 3: Selective prediction - abstain on low confidence, coverage vs macro-F1
# Standalone. Colab: best on a T4 (~10 min), also runs on CPU (~60 min).
# Reuses the v8 caches (train_sample50000 / test_full / train_blocks / test_blocks).
# Output: MyDrive/CICIoMT2024/results_v10a/
# =====================================================================

# ==== CELL 1: Setup ====
import sys, os, time, re, glob, subprocess, warnings
warnings.filterwarnings('ignore')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost', 'pyarrow'])
import numpy as np, pandas as pd, xgboost as xgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT = '/content/drive/MyDrive/CICIoMT2024'
except ImportError:
    DATA_ROOT = os.environ.get('DATA_ROOT', r'E:\CICIoMT2024')

TRAIN_DIR = os.path.join(DATA_ROOT, 'train'); TEST_DIR = os.path.join(DATA_ROOT, 'test')
CACHE_DIR = os.path.join(DATA_ROOT, 'cache'); OUT_DIR = os.path.join(DATA_ROOT, 'results_v10a')
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

try:
    import torch; HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
DEPTH, NTREES = (8, 1000) if HAS_GPU else (6, 300)          # CPU par Lite-jaisa (v9b: same accuracy)
SMOKE = os.environ.get('SMOKE') == '1'                       # small run, only to check the code
if SMOKE: NTREES = 15
K, BLOCK, CAP_TRAIN, SEED = 10, 1000, 60_000, 42
L_TEST, L_TRAIN = [5, 10, 25, 50, 100, 500], [5, 10, 25, 50, 100, 500, 1000]
REDUNDANT, SHORTCUT = ['Drate', 'Srate', 'LLC'], ['IAT', 'Number', 'Weight']
BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#9a9890'

def label_from_file(path):
    name = os.path.basename(path)
    name = re.sub(r'_(train|test)\.pcap\.csv$', '', name).replace('TCP_IP-', '')
    return re.sub(r'\d+$', '', name)

def read_csv_clean(f):
    df = pd.read_csv(f); df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    return df.astype('float32')

def fit_xgb(X, y, seed=SEED):
    a, b = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=seed)
    m = xgb.XGBClassifier(n_estimators=NTREES, learning_rate=0.1, max_depth=DEPTH, subsample=0.8,
                          colsample_bytree=0.8, tree_method='hist', device='cuda' if HAS_GPU else 'cpu',
                          early_stopping_rounds=50, random_state=seed, n_jobs=-1)
    m.fit(X[a], y[a], sample_weight=compute_sample_weight('balanced', y[a]), eval_set=[(X[b], y[b])], verbose=False)
    return m

# --- Label hierarchy ---
def family6(l):                       # the 6 official categories of the CICIoMT2024 paper
    if l == 'Benign': return 'Benign'
    if l.startswith('MQTT'): return 'MQTT'
    if l.startswith('Recon'): return 'Recon'
    if 'Spoof' in l: return 'Spoofing'
    if l.startswith('DDoS'): return 'DDoS'
    if l.startswith('DoS'): return 'DoS'
    return 'Other'
def family5(l):                       # DoS + DDoS = Flooding (attacker count ignore)
    f = family6(l); return 'Flood' if f in ('DoS', 'DDoS') else f
def type_merged(l):                   # protocol + attack, DoS/DDoS merged
    return l.replace('DDoS', 'DoS')
def binary(l):
    return 'Benign' if l == 'Benign' else 'Attack'
print('GPU:', HAS_GPU, '| depth/trees:', DEPTH, NTREES, '| Data:', DATA_ROOT)

# ==== CELL 2: Data (v8 caches) ====
train_s = pd.read_parquet(os.path.join(CACHE_DIR, 'train_sample50000.parquet'))
test_s  = pd.read_parquet(os.path.join(CACHE_DIR, 'test_full.parquet'))
LABELS = sorted(train_s['label'].unique()); L2I = {l: i for i, l in enumerate(LABELS)}
LAB = np.array(LABELS)
ALL42  = [c for c in train_s.columns if c != 'label' and c not in REDUNDANT]
FREE39 = [c for c in ALL42 if c not in SHORTCUT]
ytr_s, yte_s = train_s['label'].map(L2I).values, test_s['label'].map(L2I).values

train_b = pd.read_parquet(os.path.join(CACHE_DIR, 'train_blocks.parquet'))
test_b  = pd.read_parquet(os.path.join(CACHE_DIR, 'test_blocks.parquet'))
FEATS = [c for c in train_b.columns if c not in ('label', 'block')]
assert FEATS == FREE39 or set(FEATS) == set(FREE39), 'block features != shortcut-free 39'

def blocks_dict(df):
    d = {}
    for (lab, _), g in df.groupby(['label', 'block'], sort=False):
        d.setdefault(lab, []).append(g[FEATS].values)
    return d
def add_ctx(X, k=K):
    roll = pd.DataFrame(X, columns=FEATS).rolling(k, min_periods=1)
    return np.hstack([X, roll.mean().values, roll.std().fillna(0).values]).astype('float32')
def make_stream(bd, L_choices, n_rows, seed, benign_p=0.5):
    r = np.random.default_rng(seed); attacks = [l for l in bd if l != 'Benign']
    Xs, ys, total = [], [], 0; first = list(r.permutation(list(bd)))
    while total < n_rows or first:
        lab = first.pop() if first else ('Benign' if r.random() < benign_p else attacks[r.integers(len(attacks))])
        blk = bd[lab][r.integers(len(bd[lab]))]
        L = min(int(r.choice(L_choices)), len(blk)); s = int(r.integers(0, len(blk) - L + 1))
        Xs.append(blk[s:s + L]); ys.append(np.full(L, L2I[lab])); total += L
    return np.vstack(Xs).astype('float32'), np.concatenate(ys)

r = np.random.default_rng(42); bd_base, bd_meta = {}, {}
for lab, blks in blocks_dict(train_b).items():
    idx = r.permutation(len(blks)); cut = max(1, int(round(0.7 * len(blks))))
    bd_base[lab] = [blks[i] for i in idx[:cut]]
    bd_meta[lab] = [blks[i] for i in idx[cut:]] or [blks[i] for i in idx[:cut]]
bd_te = blocks_dict(test_b)
TEST = {L: make_stream(bd_te, [L], 20_000 if SMOKE else 200_000, seed=100 + L) for L in L_TEST}   # v8 jaise hi streams
print('Static', train_s.shape, test_s.shape, '| blocks ready | classes', len(LABELS))

# ==== CELL 3: Train models (static shortcut-free + stream ensemble, seed 42) ====
t0 = time.time()
m_static = fit_xgb(train_s[FREE39].values, ytr_s)
P_static = m_static.predict_proba(test_s[FREE39].values)
print(f'static model {time.time()-t0:.0f}s'); t0 = time.time()

Xtr, ytr = make_stream(bd_base, L_TRAIN, 30_000 if SMOKE else 600_000, seed=SEED + 1)
Xva, yva = make_stream(bd_meta, L_TEST, 10_000 if SMOKE else 150_000, seed=SEED + 1000)
m_a = fit_xgb(Xtr, ytr); m_c = fit_xgb(add_ctx(Xtr), ytr)
ws = np.round(np.arange(0, 1.01, 0.1), 1)
pa_v, pc_v = m_a.predict_proba(Xva), m_c.predict_proba(add_ctx(Xva))
W = float(ws[int(np.argmax([f1_score(yva, (w * pa_v + (1 - w) * pc_v).argmax(1), average='macro') for w in ws]))])
P_stream = {L: W * m_a.predict_proba(X) + (1 - W) * m_c.predict_proba(add_ctx(X)) for L, (X, _) in TEST.items()}
print(f'stream models {time.time()-t0:.0f}s | w={W}')
np.savez_compressed(os.path.join(OUT_DIR, 'probs_seed42.npz'), labels=LAB, P_static=P_static, y_static=yte_s,
                    **{f'P_L{L}': P_stream[L] for L in L_TEST}, **{f'y_L{L}': TEST[L][1] for L in L_TEST})

# ==== CELL 4: PART 1 - Metric hierarchy + DoS/DDoS swap share ====
LEVELS = [('Binary (attack vs benign)', binary), ('6 families (official)', family6),
          ('5 families (DoS+DDoS merged)', family5), ('13 types (DoS+DDoS merged)', type_merged),
          ('19 classes (original)', lambda l: l)]

def hierarchy(y, p, setting):
    yt, yp = LAB[y], LAB[p]; rows = []
    for name, f in LEVELS:
        a, b = np.vectorize(f)(yt), np.vectorize(f)(yp)
        rows.append({'setting': setting, 'level': name, 'n_classes': len(set(a)),
                     'accuracy': accuracy_score(a, b), 'macro_f1': f1_score(a, b, average='macro')})
    err = yt != yp
    swap = err & (np.vectorize(type_merged)(yt) == np.vectorize(type_merged)(yp))
    fam_ok = err & (np.vectorize(family5)(yt) == np.vectorize(family5)(yp))
    ben = err & ((yt == 'Benign') | (yp == 'Benign'))
    share = {'setting': setting, 'errors_%_of_windows': 100 * err.mean(),
             'DoS<->DDoS same-protocol swap (% of errors)': 100 * swap.sum() / max(err.sum(), 1),
             'right family, wrong subtype (% of errors)': 100 * fam_ok.sum() / max(err.sum(), 1),
             'involves Benign (% of errors)': 100 * ben.sum() / max(err.sum(), 1)}
    return rows, share

rows1, shares = [], []
r_, s_ = hierarchy(yte_s, P_static.argmax(1), 'static official test (shortcut-free XGB)'); rows1 += r_; shares.append(s_)
for L in L_TEST:
    r_, s_ = hierarchy(TEST[L][1], P_stream[L].argmax(1), f'mixed stream L={L} (ensemble)'); rows1 += r_; shares.append(s_)
H = pd.DataFrame(rows1); S = pd.DataFrame(shares)
H.to_csv(os.path.join(OUT_DIR, 'P1_metric_hierarchy_raw.csv'), index=False)
S.to_csv(os.path.join(OUT_DIR, 'P1_error_decomposition.csv'), index=False)

H['grp'] = np.where(H['setting'].str.startswith('static'), 'Static (official test)', 'Mixed stream (mean over L)')
TABLE_H = H.groupby(['grp', 'level'], sort=False)[['accuracy', 'macro_f1']].mean().mul(100).round(2).unstack('grp')
TABLE_H = TABLE_H.reindex([n for n, _ in LEVELS]); TABLE_H.to_csv(os.path.join(OUT_DIR, 'TABLE_H_metric_hierarchy.csv'))
print('\nTABLE H - metric hierarchy (%):'); print(TABLE_H.to_string())
print('\nError decomposition (% of errors):'); print(S.round(1).to_string(index=False))

# Binary detection detail (mixed stream): attack recall, false-positive rate
det = []
for L in L_TEST:
    yt = (LAB[TEST[L][1]] != 'Benign'); yp = (LAB[P_stream[L].argmax(1)] != 'Benign')
    det.append({'L': L, 'attack_recall': recall_score(yt, yp), 'precision': precision_score(yt, yp),
                'benign_FPR': (yp & ~yt).sum() / max((~yt).sum(), 1)})
DET = pd.DataFrame(det); DET.to_csv(os.path.join(OUT_DIR, 'P1_binary_detection.csv'), index=False)
print('\nBinary detection on mixed stream:'); print(DET.round(4).to_string(index=False))

# Top confusions (static)
yt, yp = LAB[yte_s], LAB[P_static.argmax(1)]
conf = pd.Series(list(zip(yt[yt != yp], yp[yt != yp]))).value_counts().head(10)
print('\nTop-10 confusions (static, true -> predicted, count):'); print(conf.to_string())

# Figure: metric hierarchy
fig, ax = plt.subplots(figsize=(7.2, 3.4))
lv = [n for n, _ in LEVELS]; yy = np.arange(len(lv))
for j, (g, c) in enumerate([('Static (official test)', BLUE), ('Mixed stream (mean over L)', ORANGE)]):
    v = TABLE_H[('macro_f1', g)].values
    ax.barh(yy + (j - 0.5) * 0.38, v, height=0.36, color=c, label=g)
    for k_, x in enumerate(v): ax.text(x + 0.5, yy[k_] + (j - 0.5) * 0.38, f'{x:.1f}', va='center', fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels(lv, fontsize=9); ax.invert_yaxis(); ax.set_xlim(0, 105)
ax.set_xlabel('Macro-F1 (%)'); ax.legend(fontsize=8, loc='lower right', frameon=False)
ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'fig_metric_hierarchy.png'), dpi=300); plt.close()

# ==== CELL 5: PART 2 - Recording fingerprint test ====
fp_pq = os.path.join(CACHE_DIR, 'fingerprint_train_files.parquet')
if os.path.exists(fp_pq):
    FP = pd.read_parquet(fp_pq)
else:
    parts = []
    for f in sorted(glob.glob(os.path.join(TRAIN_DIR, '*.csv'))):
        d = read_csv_clean(f)
        d = d.sample(min(len(d), 20_000), random_state=42)
        d['label'] = label_from_file(f); d['file'] = os.path.basename(f); parts.append(d)
    FP = pd.concat(parts, ignore_index=True); FP.to_parquet(fp_pq, index=False)
print('fingerprint rows', FP.shape, '| files', FP['file'].nunique())

def file_id_acc(df, feats, per_file=5000):
    df = df.groupby('file', group_keys=False).apply(lambda g: g.sample(min(len(g), per_file), random_state=42))
    f2i = {f: i for i, f in enumerate(sorted(df['file'].unique()))}
    y = df['file'].map(f2i).values
    if len(f2i) < 2: return np.nan, np.nan
    a, b = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=42)
    m = xgb.XGBClassifier(n_estimators=15 if SMOKE else 100, max_depth=6, learning_rate=0.3, tree_method='hist',
                          device='cuda' if HAS_GPU else 'cpu', n_jobs=-1, random_state=42)
    m.fit(df[feats].values[a], y[a], verbose=False)
    return accuracy_score(y[b], m.predict(df[feats].values[b])), 1 / len(f2i)

FEATSETS = [('IAT only', ['IAT']), ('Number + Weight', ['Number', 'Weight']),
            ('IAT + Number + Weight', SHORTCUT), ('Shortcut-free 39 (reference)', FREE39)]
rows2 = []
multi = [l for l, g in FP.groupby('label') if g['file'].nunique() >= 2]
print('Classes with >=2 train files (same attack, different recordings):', multi)
for lab in multi:                                   # same class -> only the recording differs
    g = FP[FP['label'] == lab]
    for name, feats in FEATSETS:
        acc, chance = file_id_acc(g, feats)
        rows2.append({'task': f'within {lab}', 'n_files': g['file'].nunique(), 'features': name, 'file_id_acc': acc, 'chance': chance})
for name, feats in FEATSETS:                        # saari files
    acc, chance = file_id_acc(FP, feats)
    rows2.append({'task': 'all train files', 'n_files': FP['file'].nunique(), 'features': name, 'file_id_acc': acc, 'chance': chance})
F = pd.DataFrame(rows2); F.to_csv(os.path.join(OUT_DIR, 'P2_fingerprint_raw.csv'), index=False)
F_within = F[F['task'] != 'all train files'].groupby('features', sort=False)[['file_id_acc', 'chance']].mean()
F_all = F[F['task'] == 'all train files'].set_index('features')[['file_id_acc', 'chance']]
TABLE_F = pd.concat({'within same class (mean)': F_within, 'all files': F_all}, axis=1).mul(100).round(2)
TABLE_F.to_csv(os.path.join(OUT_DIR, 'TABLE_F_fingerprint.csv'))
print('\nTABLE F - which PCAP file did this row come from? accuracy % (chance in brackets column):'); print(TABLE_F.to_string())

# Figure: artifact distributions per class (test set)
order = sorted(LABELS, key=lambda l: (family6(l), l))
fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharey=True)
for ax, col in zip(axes, SHORTCUT):
    data = [test_s.loc[test_s['label'] == l, col].sample(min(3000, (test_s['label'] == l).sum()), random_state=0).values for l in order]
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.6)
    for b in bp['boxes']: b.set(facecolor='#cde2fb', edgecolor=BLUE)
    for med in bp['medians']: med.set(color='#0d366b')
    if col == 'IAT': ax.set_xscale('symlog')
    ax.set_title(col, fontsize=11); ax.grid(axis='x', alpha=0.25); ax.spines[['top', 'right']].set_visible(False)
axes[0].set_yticks(range(1, len(order) + 1)); axes[0].set_yticklabels(order, fontsize=8)
plt.suptitle('Artifact features per class (CICIoMT2024 test set)', fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'fig_artifact_distributions.png'), dpi=300); plt.close()

# ==== CELL 6: PART 3 - Selective prediction (abstain when unsure) ====
COVS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
rows3 = []
def selective(P, y, setting):
    conf = P.max(1); pred = P.argmax(1); order_ = np.argsort(-conf)
    for c in COVS:
        keep = order_[:int(round(c * len(y)))]
        abst = np.setdiff1d(np.arange(len(y)), keep)
        dd = np.isin(np.vectorize(family6)(LAB[y[abst]]), ['DoS', 'DDoS']).mean() * 100 if len(abst) else np.nan
        rows3.append({'setting': setting, 'coverage': c, 'threshold': conf[order_[len(keep) - 1]],
                      'accuracy': accuracy_score(y[keep], pred[keep]), 'macro_f1': f1_score(y[keep], pred[keep], average='macro'),
                      'abstained_that_are_DoS/DDoS_%': dd})
selective(P_static, yte_s, 'static')
for L in L_TEST: selective(P_stream[L], TEST[L][1], f'stream L={L}')
SEL = pd.DataFrame(rows3); SEL.to_csv(os.path.join(OUT_DIR, 'P3_selective_raw.csv'), index=False)
SEL['grp'] = np.where(SEL['setting'] == 'static', 'static', 'stream (mean over L)')
TABLE_S = SEL.groupby(['grp', 'coverage'])[['accuracy', 'macro_f1', 'abstained_that_are_DoS/DDoS_%']].mean()
TABLE_S[['accuracy', 'macro_f1']] *= 100; TABLE_S = TABLE_S.round(2)
TABLE_S.to_csv(os.path.join(OUT_DIR, 'TABLE_S_selective.csv'))
print('\nTABLE S - selective prediction (%):'); print(TABLE_S.to_string())

fig, ax = plt.subplots(figsize=(5.6, 3.6))
for g, c in [('static', BLUE), ('stream (mean over L)', ORANGE)]:
    t = TABLE_S.loc[g]; ax.plot(t.index * 100, t['macro_f1'], marker='o', ms=5, lw=2, color=c, label=g)
ax.invert_xaxis(); ax.set_xlabel('Coverage (% windows auto-classified)'); ax.set_ylabel('19-class macro-F1 (%)')
ax.grid(alpha=0.25); ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'fig_selective_prediction.png'), dpi=300); plt.close()

print('\n==== v10a DONE ==== Saare outputs:', OUT_DIR)
print('Copy these tables: TABLE H, Error decomposition, Binary detection, Top-10 confusions, TABLE F, TABLE S')
