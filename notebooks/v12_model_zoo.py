# =====================================================================
# v12 - MODEL ZOO: fair comparison of 12 models on SHORTCUT-FREE features (IAT/Number/Weight removed)
#   Protocol S (static): official CICIoMT2024 train -> official test, one window at a time
#   Protocol M (mixed stream): benign/attack segments interleaved (L = 10, 100), 50% benign;
#       tabular models get window features (+ context = rolling mean/std of last K=10 windows),
#       sequence models (GRU, Transformer) see the last K=10 windows directly.
# Models: LogReg, GaussianNB, DecisionTree, RandomForest, ExtraTrees, LightGBM, XGBoost,
#         MLP, 1D-CNN (published architecture), GRU, Transformer, + our context ensemble.
# Every finished model is checkpointed to results_v12/ckpt -> re-running resumes.
# Colab T4 recommended (~45-75 min). CPU works but DL parts are slow.
# Output: MyDrive/CICIoMT2024/results_v12/  (metrics CSV, per-class F1, confusion matrices)
# =====================================================================

# ==== CELL 1: Setup ====
import sys, os, time, json, pickle, subprocess, warnings
warnings.filterwarnings('ignore')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost', 'lightgbm', 'pyarrow'])
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/CICIoMT2024'
except ImportError:
    ROOT = os.environ.get('DATA_ROOT', r'E:\CICIoMT2024')
CACHE = os.path.join(ROOT, 'cache'); OUT = os.path.join(ROOT, 'results_v12'); CK = os.path.join(OUT, 'ckpt')
os.makedirs(CK, exist_ok=True)
SMOKE = os.environ.get('SMOKE') == '1'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); GPU = DEV.type == 'cuda'
SEED, K = 42, 10
CAP = 600 if SMOKE else 20_000                    # training rows per class (static protocol)
N_TRAIN_STREAM = 30_000 if SMOKE else 300_000
N_TEST_STREAM = 10_000 if SMOKE else 200_000
EPOCHS = 2 if SMOKE else 10
REDUNDANT, SHORTCUT = ['Drate', 'Srate', 'LLC'], ['IAT', 'Number', 'Weight']
torch.manual_seed(SEED); np.random.seed(SEED)
print('GPU:', GPU, '| root:', ROOT)

# ==== CELL 2: Data ====
train_s = pd.read_parquet(os.path.join(CACHE, 'train_sample50000.parquet'))
test_s = pd.read_parquet(os.path.join(CACHE, 'test_full.parquet'))
LABELS = sorted(train_s['label'].unique()); L2I = {l: i for i, l in enumerate(LABELS)}; NC = len(LABELS)
FEATS = [c for c in train_s.columns if c != 'label' and c not in REDUNDANT + SHORTCUT]
tr = train_s.sample(frac=1.0, random_state=SEED); tr = tr[tr.groupby('label').cumcount() < CAP]
Xs_tr, ys_tr = tr[FEATS].to_numpy('float32'), tr['label'].map(L2I).to_numpy()
Xs_te, ys_te = test_s[FEATS].to_numpy('float32'), test_s['label'].map(L2I).to_numpy()
if SMOKE: Xs_te, ys_te = Xs_te[:20_000], ys_te[:20_000]

tr_b = pd.read_parquet(os.path.join(CACHE, 'train_blocks.parquet')); te_b = pd.read_parquet(os.path.join(CACHE, 'test_blocks.parquet'))
assert [c for c in tr_b.columns if c not in ('label', 'block')] == FEATS, 'block features must equal shortcut-free features'
def blocks(df):
    d = {}
    for (lab, _), g in df.groupby(['label', 'block'], sort=False): d.setdefault(lab, []).append(g[FEATS].to_numpy('float32'))
    return d
r = np.random.default_rng(42); bd_base, bd_meta = {}, {}
for lab, bl in blocks(tr_b).items():                      # same 70/30 block split as v8/v9
    idx = r.permutation(len(bl)); cut = max(1, int(round(0.7 * len(bl))))
    bd_base[lab] = [bl[i] for i in idx[:cut]]; bd_meta[lab] = [bl[i] for i in idx[cut:]] or [bl[i] for i in idx[:cut]]
bd_te = blocks(te_b)
def make_stream(bd, Ls, n, seed, benign_p=0.5):
    r = np.random.default_rng(seed); att = [l for l in bd if l != 'Benign']; Xs, ys, tot = [], [], 0; first = list(r.permutation(list(bd)))
    while tot < n or first:
        lab = first.pop() if first else ('Benign' if r.random() < benign_p else att[r.integers(len(att))])
        b = bd[lab][r.integers(len(bd[lab]))]; L = min(int(r.choice(Ls)), len(b)); s = int(r.integers(0, len(b) - L + 1))
        Xs.append(b[s:s + L]); ys.append(np.full(L, L2I[lab])); tot += L
    return np.vstack(Xs).astype('float32'), np.concatenate(ys)
def add_ctx(X):
    roll = pd.DataFrame(X).rolling(K, min_periods=1)
    return np.hstack([X, roll.mean().to_numpy(), roll.std().fillna(0).to_numpy()]).astype('float32')
def seqs(X):                                              # (n, K, f): last K windows, causal, padded with first row
    P = np.vstack([np.repeat(X[:1], K - 1, 0), X]); idx = np.arange(len(X))[:, None] + np.arange(K)[None, :]
    return P[idx]
Xm_tr, ym_tr = make_stream(bd_base, [5, 10, 25, 50, 100, 500, 1000], N_TRAIN_STREAM, SEED + 1)
Xm_va, ym_va = make_stream(bd_meta, [5, 10, 25, 50, 100, 500], N_TEST_STREAM // 2, SEED + 1000)
TEST_M = {L: make_stream(bd_te, [L], N_TEST_STREAM, seed=100 + L) for L in (10, 100)}   # same seeds as v8/v9
print(f'static train {Xs_tr.shape} test {Xs_te.shape} | stream train {Xm_tr.shape} | features {len(FEATS)} | classes {NC}')

# ==== CELL 3: Model definitions ====
def size_mb(obj):
    try: return len(pickle.dumps(obj)) / 1e6
    except Exception: return np.nan
def tab_models():
    return {
        'Logistic Regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight='balanced', n_jobs=-1)),
        'Gaussian Naive Bayes': lambda: make_pipeline(StandardScaler(), GaussianNB()),
        'Decision Tree': lambda: DecisionTreeClassifier(max_depth=20, class_weight='balanced', random_state=SEED),
        'Random Forest': lambda: RandomForestClassifier(n_estimators=100, max_depth=18, max_samples=0.5, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
        'Extra Trees': lambda: ExtraTreesClassifier(n_estimators=100, max_depth=18, bootstrap=True, max_samples=0.5, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
        'LightGBM': lambda: lgb.LGBMClassifier(n_estimators=20 if SMOKE else 400, learning_rate=0.1, num_leaves=63, class_weight='balanced',
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
        'XGBoost': lambda: xgb.XGBClassifier(n_estimators=20 if SMOKE else 400, learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                                             tree_method='hist', device='cuda' if GPU else 'cpu', random_state=SEED, n_jobs=-1),
    }
NEEDS_WEIGHT = {'XGBoost'}

class MLP(nn.Module):
    def __init__(s, f):
        super().__init__(); s.n = nn.Sequential(nn.Linear(f, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                                                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, NC))
    def forward(s, x): return s.n(x)
class CNN1D(nn.Module):                                   # published architecture (Mohammadi et al., ICIS 2024)
    def __init__(s, f):
        super().__init__(); s.c = nn.Sequential(nn.Conv1d(1, 32, 3), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(32, 64, 3), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten())
        with torch.no_grad(): d = s.c(torch.zeros(1, 1, f)).shape[1]
        s.h = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, NC))
    def forward(s, x): return s.h(s.c(x.unsqueeze(1)))
class GRUNet(nn.Module):
    def __init__(s, f):
        super().__init__(); s.g = nn.GRU(f, 96, batch_first=True); s.h = nn.Sequential(nn.Linear(96 + f, 128), nn.ReLU(), nn.Linear(128, NC))
    def forward(s, x): o, _ = s.g(x); return s.h(torch.cat([o[:, -1], x[:, -1]], 1))
class TransNet(nn.Module):
    def __init__(s, f, d=64):
        super().__init__(); s.inp = nn.Linear(f, d); s.pos = nn.Parameter(torch.zeros(1, K, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 128, 0.1, batch_first=True), 2)
        s.h = nn.Sequential(nn.Linear(2 * d, 128), nn.ReLU(), nn.Linear(128, NC))
    def forward(s, x): z = s.enc(s.inp(x) + s.pos); return s.h(torch.cat([z[:, -1], z.mean(1)], 1))
DL = {'MLP': MLP, '1D-CNN (published)': CNN1D}
SEQ = {'GRU (last 10 windows)': GRUNet, 'Transformer (last 10 windows)': TransNet}

def fit_torch(net_cls, Xtr, ytr, Xva=None):
    sc_mu, sc_sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    norm = lambda X: np.clip((X - sc_mu) / sc_sd, -20, 20).astype('float32')
    net = net_cls(Xtr.shape[-1]).to(DEV)
    w = torch.tensor(compute_sample_weight('balanced', ytr), dtype=torch.float32)
    cw = torch.tensor(np.bincount(ytr, minlength=NC), dtype=torch.float32).clamp(min=1); cw = (cw.sum() / (NC * cw)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); lossf = nn.CrossEntropyLoss(weight=cw)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * (len(ytr) // 1024 + 1))
    Xt, yt = torch.tensor(norm(Xtr)), torch.tensor(ytr)
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 1024):
            b = perm[i:i + 1024]; xb, yb = Xt[b].to(DEV), yt[b].to(DEV)
            loss = lossf(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    def proba(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192): out.append(torch.softmax(net(torch.tensor(norm(X[i:i + 8192])).to(DEV)), 1).cpu().numpy())
        return np.vstack(out)
    return net, proba

# ==== CELL 4: Evaluation helpers ====
ROWS, PC, CM = [], {}, {}
def evaluate(name, protocol, view, P, y, t_train, t_inf_per_1k, mb):
    pred = P.argmax(1)
    try: auc = roc_auc_score(y, P, multi_class='ovr', average='macro', labels=list(range(NC)))
    except Exception: auc = np.nan
    row = {'model': name, 'protocol': protocol, 'view': view, 'accuracy': accuracy_score(y, pred) * 100,
           'macro_f1': f1_score(y, pred, average='macro') * 100, 'weighted_f1': f1_score(y, pred, average='weighted') * 100,
           'macro_precision': precision_score(y, pred, average='macro', zero_division=0) * 100,
           'macro_recall': recall_score(y, pred, average='macro', zero_division=0) * 100, 'roc_auc_macro_ovr': auc * 100,
           'train_s': t_train, 'infer_ms_per_1k': t_inf_per_1k, 'size_mb': mb}
    pcf = f1_score(y, pred, average=None, labels=list(range(NC)), zero_division=0) * 100
    return row, pcf, confusion_matrix(y, pred, labels=list(range(NC)))
def timed_proba(fn, X):
    t0 = time.time(); P = fn(X); return P, (time.time() - t0) / len(X) * 1e6
def run(tag, fn):
    """fn() -> list of (row, perclass, cm). Checkpointed by tag."""
    ck = os.path.join(CK, tag + '.pkl')
    if os.path.exists(ck):
        res = pickle.load(open(ck, 'rb')); print(f'  {tag}: loaded checkpoint')
    else:
        t0 = time.time(); res = fn(); pickle.dump(res, open(ck, 'wb')); print(f'  {tag}: done in {time.time()-t0:.0f}s')
    for row, pcf, cm in res:
        ROWS.append(row); key = f"{row['model']} | {row['protocol']} | {row['view']}"; PC[key] = pcf; CM[key] = cm
        print(f"    {key:<70} acc {row['accuracy']:6.2f}  macro-F1 {row['macro_f1']:6.2f}")

# ==== CELL 5: Protocol S - static official split ====
print('\n=== Protocol S: official train -> official test (per window) ===')
for name, mk in tab_models().items():
    def f(name=name, mk=mk):
        m = mk(); t0 = time.time()
        if name in NEEDS_WEIGHT: m.fit(Xs_tr, ys_tr, sample_weight=compute_sample_weight('balanced', ys_tr))
        else: m.fit(Xs_tr, ys_tr)
        tt = time.time() - t0; P, inf = timed_proba(m.predict_proba, Xs_te)
        return [evaluate(name, 'S static', 'window', P, ys_te, tt, inf, size_mb(m))]
    run(f'S_{name}', f)
for name, cls in DL.items():
    def f(name=name, cls=cls):
        t0 = time.time(); net, proba = fit_torch(cls, Xs_tr, ys_tr); tt = time.time() - t0
        P, inf = timed_proba(proba, Xs_te)
        return [evaluate(name, 'S static', 'window', P, ys_te, tt, inf, sum(p.numel() for p in net.parameters()) * 4 / 1e6)]
    run(f'S_{name}', f)

# ==== CELL 6: Protocol M - mixed stream ====
print('\n=== Protocol M: mixed benign/attack stream (L = 10, 100) ===')
Xm_trc, Xm_vac = add_ctx(Xm_tr), add_ctx(Xm_va)
TEST_MC = {L: add_ctx(X) for L, (X, _) in TEST_M.items()}
FAST = ['Logistic Regression', 'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Extra Trees', 'LightGBM', 'XGBoost']
for name in FAST:
    for view, Xtr, test in [('window', Xm_tr, {L: X for L, (X, _) in TEST_M.items()}), ('window + context', Xm_trc, TEST_MC)]:
        def f(name=name, view=view, Xtr=Xtr, test=test):
            m = tab_models()[name](); t0 = time.time()
            if name in NEEDS_WEIGHT: m.fit(Xtr, ym_tr, sample_weight=compute_sample_weight('balanced', ym_tr))
            else: m.fit(Xtr, ym_tr)
            tt = time.time() - t0; out = []
            for L, X in test.items():
                P, inf = timed_proba(m.predict_proba, X); out.append(evaluate(name, f'M stream L={L}', view, P, TEST_M[L][1], tt, inf, size_mb(m)))
            return out
        run(f'M_{name}_{view}', f)
for name, cls in DL.items():
    for view, Xtr, test in [('window', Xm_tr, {L: X for L, (X, _) in TEST_M.items()}), ('window + context', Xm_trc, TEST_MC)]:
        def f(name=name, cls=cls, view=view, Xtr=Xtr, test=test):
            t0 = time.time(); net, proba = fit_torch(cls, Xtr, ym_tr); tt = time.time() - t0; out = []
            for L, X in test.items():
                P, inf = timed_proba(proba, X)
                out.append(evaluate(name, f'M stream L={L}', view, P, TEST_M[L][1], tt, inf, sum(p.numel() for p in net.parameters()) * 4 / 1e6))
            return out
        run(f'M_{name}_{view}', f)
for name, cls in SEQ.items():
    def f(name=name, cls=cls):
        t0 = time.time(); net, proba = fit_torch(cls, seqs(Xm_tr), ym_tr); tt = time.time() - t0; out = []
        for L, (X, y) in TEST_M.items():
            P, inf = timed_proba(proba, seqs(X))
            out.append(evaluate(name, f'M stream L={L}', 'sequence (K=10)', P, y, tt, inf, sum(p.numel() for p in net.parameters()) * 4 / 1e6))
        return out
    run(f'M_{name}', f)
def ens():                                                 # our method: XGB(window) + XGB(context), w tuned on validation
    mk = tab_models()['XGBoost']; a, c = mk(), mk(); t0 = time.time()
    sw = compute_sample_weight('balanced', ym_tr); a.fit(Xm_tr, ym_tr, sample_weight=sw); c.fit(Xm_trc, ym_tr, sample_weight=sw)
    tt = time.time() - t0; pa, pc = a.predict_proba(Xm_va), c.predict_proba(Xm_vac)
    ws = np.round(np.arange(0, 1.01, 0.1), 1); w = float(ws[np.argmax([f1_score(ym_va, (w * pa + (1 - w) * pc).argmax(1), average='macro') for w in ws])])
    out = []
    for L, (X, y) in TEST_M.items():
        t1 = time.time(); P = w * a.predict_proba(X) + (1 - w) * c.predict_proba(TEST_MC[L]); inf = (time.time() - t1) / len(X) * 1e6
        out.append(evaluate(f'Context ensemble (ours, w={w})', f'M stream L={L}', 'window + context', P, y, tt, inf, size_mb(a) + size_mb(c)))
    return out
run('M_ensemble', ens)

# ==== CELL 7: Tables ====
R = pd.DataFrame(ROWS).round(2); R.to_csv(os.path.join(OUT, 'zoo_all_runs.csv'), index=False)
pd.DataFrame(PC, index=LABELS).T.round(2).to_csv(os.path.join(OUT, 'zoo_per_class_f1.csv'))
np.savez_compressed(os.path.join(OUT, 'zoo_confusion.npz'), labels=np.array(LABELS), keys=np.array(list(CM)), cms=np.stack(list(CM.values())))
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
cols = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall', 'roc_auc_macro_ovr', 'train_s', 'infer_ms_per_1k', 'size_mb']
print('\nTABLE Z1 - Protocol S (official test, shortcut-free features):')
print(R[R['protocol'] == 'S static'].sort_values('macro_f1', ascending=False)[['model'] + cols].to_string(index=False))
M = R[R['protocol'].str.startswith('M')].groupby(['model', 'view'], sort=False)[cols].mean().round(2).sort_values('macro_f1', ascending=False)
M.to_csv(os.path.join(OUT, 'zoo_stream_mean.csv'))
print('\nTABLE Z2 - Protocol M (mixed stream, mean of L=10 and L=100):'); print(M.to_string())
print('\n==== v12 DONE ==== Output:', OUT, '\nPaste TABLE Z1 and TABLE Z2.')
