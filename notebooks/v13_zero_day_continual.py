# =====================================================================
# v13 - ZERO-DAY (open-set) DETECTION + PATTERN DISCOVERY + CONTINUAL LEARNING
#   For each unseen-attack scenario, one attack group is removed from training entirely.
#   Z1  Unknown scores on the 19-slot window model: MSP, energy, kNN distance, IsolationForest,
#       combined (kNN + energy), and kNN WITH the IAT timestamp (ablation).
#       Thresholds are set on KNOWN validation windows only (5% known false-unknown target).
#   Z2  System view: is an unseen attack window alerted (as some known attack) or flagged unknown?
#   Z3  Pattern discovery: KMeans on flagged-unknown windows, k picked by silhouette -> purity / ARI.
#   Z4  Continual learning after an analyst labels n windows per new class:
#       A fine-tune WITHOUT replay | B fine-tune WITH replay | C retrain on replay buffer + new
#       | D full retrain (upper bound). Old-class F1 (forgetting) and new-class F1 on the official test set.
#   EXPORT reference_train.parquet / reference_val.parquet for the project (shortcut-free features).
# Every scenario is checkpointed to results_v13/ckpt -> re-running resumes.
# Colab T4 ~35-50 min | CPU ~1.5-2.5 h
# Output: MyDrive/CICIoMT2024/results_v13/
# =====================================================================

# ==== CELL 1: Setup ====
import sys, os, time, json, subprocess, warnings
warnings.filterwarnings('ignore')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost', 'pyarrow'])
import numpy as np, pandas as pd, xgboost as xgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score, silhouette_score, adjusted_rand_score
from sklearn.utils.class_weight import compute_sample_weight
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/CICIoMT2024'
except ImportError:
    ROOT = os.environ.get('DATA_ROOT', r'E:\CICIoMT2024')
CACHE = os.path.join(ROOT, 'cache'); OUT = os.path.join(ROOT, 'results_v13'); CK = os.path.join(OUT, 'ckpt')
os.makedirs(CK, exist_ok=True)
try:
    import torch; GPU = torch.cuda.is_available()
except ImportError:
    GPU = False
SMOKE = os.environ.get('SMOKE') == '1'
DEV = 'cuda' if GPU else 'cpu'
SEED = 42
N_VAL, N_REF = (100, 100) if SMOKE else (2000, 1000)      # per class: threshold-validation rows, reference/replay rows
CAP = 300 if SMOKE else 10_000                            # base training rows per class
TEST_CAP_KNOWN, TEST_CAP_UNK = (200, 400) if SMOKE else (2000, 5000)   # per class
TREES = 20 if SMOKE else 300
N_LABEL = [10, 50] if SMOKE else [10, 50, 200, 1000]     # analyst-labelled windows per new class
FT_ROUNDS = 40 if SMOKE else 60
KNN_K = 10
TARGET_FPR = 0.05
REDUNDANT, SHORTCUT = ['Drate', 'Srate', 'LLC'], ['IAT', 'Number', 'Weight']
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#9a9890'
print('GPU:', GPU, '| root:', ROOT, '| smoke:', SMOKE)

# ==== CELL 2: Data + splits ====
train_s = pd.read_parquet(os.path.join(CACHE, 'train_sample50000.parquet'))
test_s = pd.read_parquet(os.path.join(CACHE, 'test_full.parquet'))
LABELS = sorted(train_s['label'].unique()); L2I = {l: i for i, l in enumerate(LABELS)}; NC = len(LABELS)
BENIGN = L2I['Benign']
FEATS = [c for c in train_s.columns if c != 'label' and c not in REDUNDANT + SHORTCUT]
FEATS_IAT = FEATS + ['IAT']
sh = train_s.sample(frac=1.0, random_state=SEED); rank = sh.groupby('label').cumcount()
VAL = sh[rank < N_VAL]                                     # never trained on -> thresholds
REF = sh[(rank >= N_VAL) & (rank < N_VAL + N_REF)]         # reference set + replay buffer (also part of training)
TRN = sh[(rank >= N_VAL) & (rank < N_VAL + CAP)]           # base training (includes REF)
POOL = sh[rank >= N_VAL + CAP]                             # analyst-label pool for continual learning (unused rows)
if POOL['label'].value_counts().reindex(LABELS, fill_value=0).min() < max(N_LABEL):
    POOL = TRN                                             # small classes: label pool falls back to training rows
te = test_s.sample(frac=1.0, random_state=SEED)
print(f'classes {NC} | features {len(FEATS)} | train {len(TRN):,} | val {len(VAL):,} | ref {len(REF):,} | test {len(te):,}')

SCENARIOS = {                                              # unseen-attack scenario -> held-out classes
    'ARP spoofing': ['ARP_Spoofing'],
    'Reconnaissance (4)': [l for l in LABELS if l.startswith('Recon')],
    'MQTT attacks (5)': [l for l in LABELS if l.startswith('MQTT')],
    'MQTT malformed data': ['MQTT-Malformed_Data'],
    'ICMP floods (DoS+DDoS)': [l for l in LABELS if l.endswith('-ICMP')],
    'SYN floods (DoS+DDoS)': [l for l in LABELS if l.endswith('-SYN')],
    'UDP floods (DoS+DDoS)': [l for l in LABELS if l.endswith('-UDP')],
    'TCP floods (DoS+DDoS)': [l for l in LABELS if l.endswith('-TCP')],
}
for k, v in SCENARIOS.items(): assert v and all(l in L2I for l in v), (k, v)

# ==== CELL 3: Helpers ====
def params(seed=SEED):
    return {'objective': 'multi:softprob', 'num_class': NC, 'eta': 0.1, 'max_depth': 6, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'tree_method': 'hist', 'device': DEV, 'seed': seed, 'eval_metric': 'mlogloss'}
def dmat(df, feats=FEATS, weight=True):
    y = df['label'].map(L2I).to_numpy()
    return xgb.DMatrix(df[feats].to_numpy('float32'), label=y, weight=compute_sample_weight('balanced', y) if weight else None)
def train_booster(df, rounds=TREES, base=None, feats=FEATS):
    """19-slot booster: held-out classes simply have no rows. base -> continue boosting (fine-tune)."""
    if base is None:
        u = np.random.default_rng(SEED).random(len(df)); a, b = df[u < 0.85], df[u >= 0.85]
        return xgb.train(params(), dmat(a, feats), rounds, evals=[(dmat(b, feats), 'val')], early_stopping_rounds=30, verbose_eval=False)
    # A class absent from base training has P ~ 0 -> hessian ~ 0, so default settings can never grow its trees.
    # Unblock it: no min hessian, ~no L2, and max_delta_step bounds each leaf step.
    p = params(); p.update({'eta': 0.3, 'lambda': 1e-3, 'min_child_weight': 0, 'max_delta_step': 2})
    return xgb.train(p, dmat(df, feats), rounds, xgb_model=base, verbose_eval=False)
def proba(bst, X): return bst.inplace_predict(X)
def margin(bst, X): return bst.inplace_predict(X, predict_type='margin')

class Space:                                               # signed log1p + z-score, fitted on the reference set
    def __init__(self, X):
        Z = self._t(X); self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
    @staticmethod
    def _t(X): return np.sign(X) * np.log1p(np.abs(X))
    def __call__(self, X): return ((self._t(X) - self.mu) / self.sd).astype('float32')

def knn_score(ref_X, Xq_list):
    sp = Space(ref_X); nn_ = NearestNeighbors(n_neighbors=KNN_K).fit(sp(ref_X))
    return [nn_.kneighbors(sp(X))[0].mean(1) for X in Xq_list]

def cap(df, n, seed=SEED): return df[df.groupby('label').cumcount() < n] if n else df
def macro_f1(y, p, classes): return f1_score(y, p, labels=classes, average='macro', zero_division=0) * 100

# ==== CELL 4: Z1-Z3 per scenario ====
def run_scenario(name, held):
    ck = os.path.join(CK, f"zd_{name.split(' ')[0]}_{len(held)}.json")
    if os.path.exists(ck): return json.load(open(ck))
    t0 = time.time(); held_i = [L2I[l] for l in held]; known_i = [i for i in range(NC) if i not in held_i]
    trn, val, ref = TRN[~TRN['label'].isin(held)], VAL[~VAL['label'].isin(held)], REF[~REF['label'].isin(held)]
    te_k = cap(te[~te['label'].isin(held)], TEST_CAP_KNOWN); te_u = cap(te[te['label'].isin(held)], TEST_CAP_UNK)
    bst = train_booster(trn)
    Xv, Xk, Xu = (d[FEATS].to_numpy('float32') for d in (val, te_k, te_u))
    S = {}
    for nm, X in (('val', Xv), ('known', Xk), ('unk', Xu)):
        P = proba(bst, X)[:, known_i]; P = P / P.sum(1, keepdims=True); Mg = margin(bst, X)[:, known_i]
        S.setdefault('MSP', {})[nm] = 1 - P.max(1)
        S.setdefault('Energy', {})[nm] = -logsumexp(Mg, axis=1)
        S.setdefault('_patt', {})[nm] = 1 - P[:, known_i.index(BENIGN)]
        S.setdefault('_pred', {})[nm] = np.array(known_i)[P.argmax(1)]
    for nm, s in zip(('val', 'known', 'unk'), knn_score(ref[FEATS].to_numpy('float32'), [Xv, Xk, Xu])): S.setdefault('kNN distance', {})[nm] = s
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Space(ref[FEATS].to_numpy('float32'))(ref[FEATS].to_numpy('float32')))
    spf = Space(ref[FEATS].to_numpy('float32'))
    for nm, X in (('val', Xv), ('known', Xk), ('unk', Xu)): S.setdefault('IsolationForest', {})[nm] = -iso.score_samples(spf(X))
    def ecdf(ref_scores, s): return np.searchsorted(np.sort(ref_scores), s, side='right') / len(ref_scores)
    S['kNN + Energy'] = {nm: np.maximum(ecdf(S['kNN distance']['val'], S['kNN distance'][nm]), ecdf(S['Energy']['val'], S['Energy'][nm]))
                         for nm in ('val', 'known', 'unk')}
    if all(c in TRN.columns for c in FEATS_IAT):            # ablation: same kNN but the IAT timestamp is kept
        for nm, s in zip(('val', 'known', 'unk'), knn_score(ref[FEATS_IAT].to_numpy('float32'),
                                                            [d[FEATS_IAT].to_numpy('float32') for d in (val, te_k, te_u)])):
            S.setdefault('kNN distance (with IAT)', {})[nm] = s
    yk = te_k['label'].map(L2I).to_numpy(); ben = yk == BENIGN
    rows, best = [], None
    for sc, d in S.items():
        if sc.startswith('_'): continue
        thr = np.quantile(d['val'], 1 - TARGET_FPR)
        r = {'scenario': name, 'score': sc, 'AUROC_%': roc_auc_score(np.r_[np.zeros(len(d['known'])), np.ones(len(d['unk']))],
                                                                    np.r_[d['known'], d['unk']]) * 100,
             'unknown_recall_%': (d['unk'] >= thr).mean() * 100, 'known_test_false_unknown_%': (d['known'] >= thr).mean() * 100,
             'benign_false_unknown_%': (d['known'][ben] >= thr).mean() * 100}
        rows.append(r)
    # Z2 system view with the combined score + validated P(attack) >= 0.95 rule
    thr_c = np.quantile(S['kNN + Energy']['val'], 1 - TARGET_FPR)
    unk_flag, att_flag = S['kNN + Energy']['unk'] >= thr_c, S['_patt']['unk'] >= 0.95
    sysrow = {'scenario': name, 'held_out': ', '.join(held), 'n_unseen_test_windows': int(len(Xu)),
              'alerted_as_known_attack_%': att_flag.mean() * 100, 'flagged_unknown_%': unk_flag.mean() * 100,
              'caught_total_%': (att_flag | unk_flag).mean() * 100, 'missed_%': (~(att_flag | unk_flag)).mean() * 100,
              'benign_flagged_unknown_%': (S['kNN + Energy']['known'][ben] >= thr_c).mean() * 100,
              'known_attack_type_accuracy_%': (S['_pred']['known'] == yk).mean() * 100}
    # Z3 pattern discovery on the flagged-unknown unseen windows
    Xf, yf = Xu[unk_flag | att_flag], te_u['label'].to_numpy()[unk_flag | att_flag]
    patt = {'scenario': name, 'true_k': len(held), 'n_clustered': int(len(Xf))}
    if len(Xf) >= 50:
        Z = spf(Xf); sub = np.random.default_rng(0).choice(len(Z), min(len(Z), 3000), replace=False)
        ks = [k for k in range(2, 9) if k < len(sub)]
        sil = {k: silhouette_score(Z[sub], KMeans(k, n_init=5, random_state=0).fit_predict(Z[sub])) for k in ks}
        kbest = max(sil, key=sil.get) if len(held) > 1 else 1
        if len(held) == 1 and max(sil.values()) > 0.5: kbest = max(sil, key=sil.get)
        lab = KMeans(kbest, n_init=5, random_state=0).fit_predict(Z) if kbest > 1 else np.zeros(len(Z), int)
        ct = pd.crosstab(lab, yf)
        patt.update({'chosen_k': int(kbest), 'purity_%': ct.max(1).sum() / ct.values.sum() * 100,
                     'ARI': adjusted_rand_score(yf, lab) if len(held) > 1 else np.nan})
    res = {'scores': rows, 'system': sysrow, 'patterns': patt, 'sec': time.time() - t0}
    json.dump(res, open(ck, 'w'), default=float)
    return res

ZD = {}
for name, held in SCENARIOS.items():
    ZD[name] = run_scenario(name, held); print(f"Z {name:26s} done ({ZD[name]['sec']:.0f}s)")
Z1 = pd.DataFrame([r for v in ZD.values() for r in v['scores']]); Z1.round(2).to_csv(os.path.join(OUT, 'Z1_unknown_scores.csv'), index=False)
Z2 = pd.DataFrame([v['system'] for v in ZD.values()]); Z2.round(2).to_csv(os.path.join(OUT, 'Z2_system_view.csv'), index=False)
Z3 = pd.DataFrame([v['patterns'] for v in ZD.values()]); Z3.round(3).to_csv(os.path.join(OUT, 'Z3_pattern_discovery.csv'), index=False)
pd.set_option('display.width', 220)
print('\nZ1 - AUROC (%) known vs unseen, per score:'); print(Z1.pivot(index='scenario', columns='score', values='AUROC_%').round(1).to_string())
print('\nZ1 - unseen recall at 5% validation false-unknown rate:'); print(Z1.pivot(index='scenario', columns='score', values='unknown_recall_%').round(1).to_string())
print('\nZ1 - false-unknown rate on the KNOWN official test set (target 5%):'); print(Z1.pivot(index='scenario', columns='score', values='known_test_false_unknown_%').round(1).to_string())
print('\nZ2 - system view (combined score + P(attack) >= 0.95):'); print(Z2.drop(columns=['held_out']).round(2).to_string(index=False))
print('\nZ3 - pattern discovery:'); print(Z3.round(3).to_string(index=False))

# ==== CELL 5: Z4 continual learning ====
def cl_scenario(name, held):
    ck = os.path.join(CK, f"cl_{name.split(' ')[0]}_{len(held)}.csv")
    if os.path.exists(ck): return pd.read_csv(ck)
    held_i = [L2I[l] for l in held]; known_i = [i for i in range(NC) if i not in held_i]
    trn = TRN[~TRN['label'].isin(held)]; ref = REF[~REF['label'].isin(held)]
    ev = cap(te, TEST_CAP_KNOWN); Xe, ye = ev[FEATS].to_numpy('float32'), ev['label'].map(L2I).to_numpy()
    def score(bst, strat, n, sec):
        p = proba(bst, Xe).argmax(1)
        return {'scenario': name, 'strategy': strat, 'n_per_class': n, 'old_macro_f1': macro_f1(ye, p, known_i),
                'new_macro_f1': macro_f1(ye, p, held_i), 'all19_macro_f1': macro_f1(ye, p, list(range(NC))), 'train_sec': sec}
    t = time.time(); base = train_booster(trn); rows = [score(base, '0 before learning', 0, time.time() - t)]
    pool = POOL[POOL['label'].isin(held)].sample(frac=1.0, random_state=1)
    for n in N_LABEL:
        new = cap(pool, n)
        for strat, data, kind in [('A fine-tune, no replay', new, 'ft'), ('B fine-tune + replay', pd.concat([cap(ref, 200), new]), 'ft'),
                                  ('C retrain on replay buffer', pd.concat([ref, new]), 'scratch')]:
            t = time.time(); bst = train_booster(data, FT_ROUNDS, base=base) if kind == 'ft' else train_booster(data)
            rows.append(score(bst, strat, n, time.time() - t))
    t = time.time(); full = train_booster(pd.concat([trn, cap(pool, max(N_LABEL))]))
    rows.append(score(full, 'D full retrain (upper bound)', max(N_LABEL), time.time() - t))
    out = pd.DataFrame(rows); base_old = out.iloc[0]['old_macro_f1']; out['forgetting_pts'] = base_old - out['old_macro_f1']
    out.to_csv(ck, index=False); return out

CL = []
for name, held in SCENARIOS.items():
    CL.append(cl_scenario(name, held)); print(f'CL {name:26s} done')
CL = pd.concat(CL); CL.round(2).to_csv(os.path.join(OUT, 'Z4_continual_learning.csv'), index=False)
agg = CL.groupby(['strategy', 'n_per_class'])[['old_macro_f1', 'new_macro_f1', 'all19_macro_f1', 'forgetting_pts', 'train_sec']].mean().round(2)
print('\nZ4 - continual learning (mean over scenarios, official test set):'); print(agg.to_string())

# ==== CELL 6: Figures ====
fig, ax = plt.subplots(figsize=(8.6, 3.8))
order = ['MSP', 'Energy', 'IsolationForest', 'kNN distance', 'kNN + Energy']
pv = Z1.pivot(index='scenario', columns='score', values='AUROC_%').reindex(columns=order)
x = np.arange(len(pv)); w = 0.16
for i, (sc, c) in enumerate(zip(order, [GREY, YELLOW, MAGENTA, BLUE, ORANGE])):
    ax.bar(x + (i - 2) * w, pv[sc], w * 0.9, color=c, label=sc)
ax.set_xticks(x); ax.set_xticklabels(pv.index, rotation=25, ha='right', fontsize=8); ax.set_ylabel('AUROC known vs unseen (%)')
ax.set_ylim(0, 100.5); ax.axhline(50, color=GREY, lw=0.8, ls='--'); ax.legend(frameon=False, fontsize=7, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.13))
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_zero_day_auroc.png'), dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4), sharex=True)
cols = {'A fine-tune, no replay': MAGENTA, 'B fine-tune + replay': BLUE, 'C retrain on replay buffer': ORANGE}
a2 = CL[CL['strategy'].isin(cols)].groupby(['strategy', 'n_per_class'])[['old_macro_f1', 'new_macro_f1']].mean().reset_index()
b0 = CL[CL['strategy'].str.startswith('0')][['old_macro_f1', 'new_macro_f1']].mean(); up = CL[CL['strategy'].str.startswith('D')][['old_macro_f1', 'new_macro_f1']].mean()
for ax, m, t in zip(axs, ['new_macro_f1', 'old_macro_f1'], ['New (previously unseen) classes', 'Old classes (forgetting check)']):
    for s, c in cols.items():
        d = a2[a2['strategy'] == s]; ax.plot(d['n_per_class'], d[m], marker='o', ms=5, lw=2, color=c, label=s)
    ax.axhline(up[m], color=AQUA, lw=1.2, ls='--', label='full retrain (upper bound)'); ax.axhline(b0[m], color=GREY, lw=1, ls=':', label='before learning')
    ax.set_xscale('log'); ax.set_xticks(N_LABEL); ax.set_xticklabels(N_LABEL); ax.minorticks_off(); ax.set_title(t, fontsize=9); ax.set_xlabel('labelled windows per new class'); ax.set_ylabel('macro-F1 (%)')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7, ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, 'fig_continual_learning.png'), dpi=200); plt.close(fig)

# ==== CELL 7: Export reference data for the project (copy to E:\iomt-ids\data\) ====
REF[FEATS + ['label']].reset_index(drop=True).to_parquet(os.path.join(OUT, 'reference_train.parquet'), index=False)
cap(VAL, 500)[FEATS + ['label']].reset_index(drop=True).to_parquet(os.path.join(OUT, 'reference_val.parquet'), index=False)
print('\nExported reference_train.parquet', REF[FEATS + ['label']].shape, 'and reference_val.parquet ->', OUT)
print('Copy to the project data/ folder: reference_train.parquet, reference_val.parquet,')
print('  Z1_unknown_scores.csv, Z2_system_view.csv, Z3_pattern_discovery.csv, Z4_continual_learning.csv, fig_*.png  (into data/zeroday/)')
