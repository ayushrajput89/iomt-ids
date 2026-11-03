# =====================================================================
# CICIoMT2024 - v10b: XAI as a shortcut detector (Idea 2)
#   1. Global SHAP: all-42-feature model vs shortcut-free-39 model (3 seeds)
#      -> how much of the attribution goes to IAT/Number/Weight?
#   2. SHAP stability across seeds (Spearman rank correlation, top-10 overlap)
#   3. Per-family top features of the shortcut-free model (what it really uses)
# Standalone, CPU OK (~15-25 min). Uses cache/train_sample50000 + test_full.
# Output: MyDrive/CICIoMT2024/results_v10b/
# =====================================================================

# ==== CELL 1: Setup + data ====
import sys, os, time, itertools, subprocess, warnings
warnings.filterwarnings('ignore')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost', 'pyarrow'])
import numpy as np, pandas as pd, xgboost as xgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
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
CACHE_DIR = os.path.join(DATA_ROOT, 'cache'); OUT_DIR = os.path.join(DATA_ROOT, 'results_v10b')
os.makedirs(OUT_DIR, exist_ok=True)
try:
    import torch; HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
DEPTH, NTREES = (8, 1000) if HAS_GPU else (6, 300)
if os.environ.get('SMOKE') == '1': NTREES = 15
SEEDS, N_EXPLAIN = [42, 7, 123], 5000
REDUNDANT, SHORTCUT = ['Drate', 'Srate', 'LLC'], ['IAT', 'Number', 'Weight']
BLUE, ORANGE = '#2a78d6', '#eb6834'

def family6(l):
    if l == 'Benign': return 'Benign'
    if l.startswith('MQTT'): return 'MQTT'
    if l.startswith('Recon'): return 'Recon'
    if 'Spoof' in l: return 'Spoofing'
    if l.startswith('DDoS'): return 'DDoS'
    return 'DoS'

def fit_xgb(X, y, seed):
    a, b = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=seed)
    m = xgb.XGBClassifier(n_estimators=NTREES, learning_rate=0.1, max_depth=DEPTH, subsample=0.8,
                          colsample_bytree=0.8, tree_method='hist', device='cuda' if HAS_GPU else 'cpu',
                          early_stopping_rounds=50, random_state=seed, n_jobs=-1)
    m.fit(X[a], y[a], sample_weight=compute_sample_weight('balanced', y[a]), eval_set=[(X[b], y[b])], verbose=False)
    return m

train_s = pd.read_parquet(os.path.join(CACHE_DIR, 'train_sample50000.parquet'))
test_s  = pd.read_parquet(os.path.join(CACHE_DIR, 'test_full.parquet'))
LABELS = sorted(train_s['label'].unique()); L2I = {l: i for i, l in enumerate(LABELS)}
ALL42  = [c for c in train_s.columns if c != 'label' and c not in REDUNDANT]
FREE39 = [c for c in ALL42 if c not in SHORTCUT]
ytr, yte = train_s['label'].map(L2I).values, test_s['label'].map(L2I).values
# fixed explain set: class-balanced sample of the official test set (same rows for every model/seed)
EX = test_s.sample(frac=1.0, random_state=0)
EX = EX[EX.groupby('label').cumcount() < N_EXPLAIN // len(LABELS)]
yex = EX['label'].map(L2I).values
print('GPU:', HAS_GPU, '| depth/trees:', DEPTH, NTREES, '| explain rows:', len(EX))

# ==== CELL 2: Train + SHAP (2 settings x 3 seeds) ====
def shap_values(model, X):
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)   # (n, classes, f+1)
    return c[:, :, :-1]                                                    # drop bias term

rows, PERCLASS = [], {}
for seed in SEEDS:
    for setting, feats in [('All 42 features', ALL42), ('Shortcut-free 39', FREE39)]:
        t0 = time.time()
        m = fit_xgb(train_s[feats].values, ytr, seed)
        pred = m.predict(test_s[feats].values)
        sv = shap_values(m, EX[feats].values)
        glob_imp = np.abs(sv).sum(1).mean(0)                               # mean |SHAP| summed over classes
        own = sv[np.arange(len(yex)), yex, :]                              # contribution to the true class
        PERCLASS[(setting, seed)] = pd.DataFrame(np.abs(own), columns=feats).assign(label=EX['label'].values)
        for f, v in zip(feats, glob_imp):
            rows.append({'setting': setting, 'seed': seed, 'feature': f, 'mean_abs_shap': v})
        print(f'{setting} seed {seed}: acc {accuracy_score(yte, pred):.4f} macro-F1 {f1_score(yte, pred, average="macro"):.4f} | {time.time()-t0:.0f}s')
G = pd.DataFrame(rows)
G['share_%'] = 100 * G['mean_abs_shap'] / G.groupby(['setting', 'seed'])['mean_abs_shap'].transform('sum')
G.to_csv(os.path.join(OUT_DIR, 'shap_global_raw.csv'), index=False)

# ==== CELL 3: Tables ====
# (1) artifact share in the all-feature model
art = G[(G['setting'] == 'All 42 features') & G['feature'].isin(SHORTCUT)].groupby('seed')['share_%'].sum()
print(f'\nShare of total |SHAP| on IAT+Number+Weight (3 of 42 features): {art.mean():.2f} +/- {art.std():.2f} %')
per_art = G[(G['setting'] == 'All 42 features') & G['feature'].isin(SHORTCUT)].groupby('feature')['share_%'].mean().round(2)
print(per_art.to_string())

# (2) top-15 per setting (mean over seeds) + rank
TOP = G.groupby(['setting', 'feature'])['share_%'].agg(['mean', 'std']).round(2).reset_index()
TOP['rank'] = TOP.groupby('setting')['mean'].rank(ascending=False).astype(int)
TOP = TOP.sort_values(['setting', 'rank']); TOP.to_csv(os.path.join(OUT_DIR, 'TABLE_X1_shap_global.csv'), index=False)
for s in ['All 42 features', 'Shortcut-free 39']:
    print(f'\nTop-10 features, {s} (share of |SHAP| %, mean +/- std over 3 seeds):')
    print(TOP[TOP['setting'] == s].head(10)[['rank', 'feature', 'mean', 'std']].to_string(index=False))

# (3) stability across seeds
stab = []
for s in ['All 42 features', 'Shortcut-free 39']:
    P = G[G['setting'] == s].pivot(index='feature', columns='seed', values='mean_abs_shap')
    for a, b in itertools.combinations(SEEDS, 2):
        top_a, top_b = set(P[a].nlargest(10).index), set(P[b].nlargest(10).index)
        stab.append({'setting': s, 'pair': f'{a}-{b}', 'spearman': spearmanr(P[a], P[b]).correlation,
                     'top10_overlap': len(top_a & top_b) / 10})
STAB = pd.DataFrame(stab); STAB.to_csv(os.path.join(OUT_DIR, 'TABLE_X2_shap_stability.csv'), index=False)
print('\nSHAP stability across seeds (mean):'); print(STAB.groupby('setting')[['spearman', 'top10_overlap']].mean().round(3).to_string())

# (4) per-family top-5 features of the shortcut-free model (seed 42)
pc = PERCLASS[('Shortcut-free 39', 42)].copy(); pc['family'] = pc['label'].map(family6)
fam = pc.drop(columns='label').groupby('family').mean()
fam_share = fam.div(fam.sum(1), axis=0) * 100
FAMTOP = pd.DataFrame({f: [f'{c} ({fam_share.loc[f, c]:.1f}%)' for c in fam_share.loc[f].nlargest(5).index] for f in fam_share.index},
                      index=[f'#{i}' for i in range(1, 6)]).T
FAMTOP.to_csv(os.path.join(OUT_DIR, 'TABLE_X3_family_top_features.csv'))
print('\nShortcut-free model - top-5 features per attack family (share of true-class |SHAP|):'); print(FAMTOP.to_string())

# ==== CELL 4: Figure ====
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, s in zip(axes, ['All 42 features', 'Shortcut-free 39']):
    t = TOP[TOP['setting'] == s].head(15).iloc[::-1]
    cols = [ORANGE if f in SHORTCUT else BLUE for f in t['feature']]
    ax.barh(t['feature'], t['mean'], xerr=t['std'], color=cols, height=0.65, error_kw={'elinewidth': 0.8, 'ecolor': '#555'})
    for yv, (v, sd) in enumerate(zip(t['mean'], t['std'].fillna(0))): ax.text(v + sd + 0.02 * t['mean'].max(), yv, f'{v:.1f}', va='center', fontsize=8)
    ax.set_title(s, fontsize=11); ax.set_xlabel('Share of mean |SHAP| (%)')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=0.25); ax.tick_params(axis='y', labelsize=8)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=ORANGE, label='Artifact feature (IAT / Number / Weight)'), Patch(color=BLUE, label='Behavioural feature')],
           loc='lower center', ncol=2, frameon=False, fontsize=9)
plt.tight_layout(rect=[0, 0.05, 1, 1]); plt.savefig(os.path.join(OUT_DIR, 'fig_shap_all_vs_shortcut_free.png'), dpi=300); plt.close()
print('\n==== v10b DONE ==== Outputs:', OUT_DIR)
print('Paste: artifact share, both top-10 tables, stability, per-family top-5')
