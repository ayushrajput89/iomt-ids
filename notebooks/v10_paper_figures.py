# =====================================================================
# v10 - ALL paper figures + tables from saved results (no training, CPU, ~2-4 min)
# Reads MyDrive/CICIoMT2024/results_v8 ... results_v10f ; writes results_v10_figures/
#   fig_XX_*.pdf (vector, for LaTeX) + fig_XX_*.png (300 dpi, for Word/preview)
#   tab_XX_*.tex (IEEE-ready LaTeX) + tab_XX_*.csv
# Each figure is independent: if one input is missing, that figure is skipped with a message.
# Frozen numbers from the research log (v8b CNN, v9/v9b system) are hard-coded and marked.
# =====================================================================
import os, json, datetime as dt, traceback, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, Patch
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/CICIoMT2024'
except ImportError:
    ROOT = os.environ.get('DATA_ROOT', r'E:\CICIoMT2024')
OUT = os.path.join(ROOT, 'results_v10_figures'); os.makedirs(OUT, exist_ok=True)
R = lambda v, f: os.path.join(ROOT, f'results_{v}', f)

# ---------- style (IEEE: single column 3.5 in, double column 7.16 in, 8 pt text) ----------
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42})
S1, S2 = 3.5, 7.16
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'
GREY, INK2 = '#9a9890', '#52514e'
FAM_COL = {'DDoS': BLUE, 'DoS': ORANGE, 'MQTT': AQUA, 'Recon': YELLOW, 'Spoofing': MAGENTA, 'Benign': GREY}
def family6(l):
    if l == 'Benign': return 'Benign'
    if l.startswith('MQTT'): return 'MQTT'
    if l.startswith('Recon'): return 'Recon'
    if 'Spoof' in l: return 'Spoofing'
    return 'DDoS' if l.startswith('DDoS') else 'DoS'
def save(fig, name):
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300)
    plt.close(fig); print('  saved', name)
def table(df, name, caption, label, index=False):
    df.to_csv(os.path.join(OUT, f'{name}.csv'), index=index)
    tex = df.to_latex(index=index, escape=True, float_format=lambda v: f'{v:.2f}', column_format=None)
    with open(os.path.join(OUT, f'{name}.tex'), 'w') as fh:
        fh.write('\\begin{table}[t]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n'
                 '\\resizebox{\\columnwidth}{!}{%\n' + tex + '}\n\\end{table}\n')
    print('  saved', name)
DONE, SKIP = [], []
def run(name, fn):
    print(f'[{name}]')
    try: fn(); DONE.append(name)
    except Exception as e:
        SKIP.append(f'{name}: {type(e).__name__}: {e}'); print('  SKIPPED ->', e)

# =====================================================================
# Fig 1 - system / methodology overview (drawn, no data)
# =====================================================================
def fig01():
    fig, ax = plt.subplots(figsize=(S2, 2.3)); ax.set_xlim(0, 104); ax.set_ylim(0, 30); ax.axis('off')
    def box(x, y, w, h, text, fc='#eef4fc', ec=BLUE, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=6.0, weight='bold' if bold else 'normal')
    def arr(x1, y1, x2, y2):
        ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK2, shrinkA=0, shrinkB=0))
    ax.text(1, 28.2, '(A) Shortcut audit', fontsize=7.5, weight='bold'); ax.text(52, 28.2, '(B) Shortcut-aware, context-aware IDS', fontsize=7.5, weight='bold')
    box(1, 18, 13, 7, 'CICIoMT2024\n19 classes\n45 features', fc='#f3f2ef', ec=GREY)
    box(17, 18, 15, 7, 'Single-feature\nscan + ablation')
    box(17, 8, 15, 7, 'SHAP +\nfingerprinting')
    box(35, 13, 14, 7, 'IAT decode:\nIAT x k =\ntimestamp', fc='#fdeee7', ec=ORANGE, bold=True)
    arr(14, 21.5, 17, 21.5); arr(14, 20, 17, 12); arr(32, 21.5, 35, 17.5); arr(32, 11.5, 35, 15.5)
    box(52, 13, 9, 7, 'Traffic\nwindow', fc='#f3f2ef', ec=GREY)
    box(63, 13, 9, 7, 'Artifact\nguard\n(-IAT)', fc='#fdeee7', ec=ORANGE)
    box(74, 18.5, 12, 6, 'Per-window\nXGBoost')
    box(74, 6.5, 12, 8, 'Context\n(K=10 mean/std)\n+ XGBoost')
    box(89, 12, 14, 9, 'Fusion\n-> threshold\n-> 3-of-5 alert\n-> SHAP reason', fc='#e8f6ef', ec=AQUA)
    arr(61, 16.5, 63, 16.5); arr(72, 17.5, 74, 21.5); arr(72, 15.5, 74, 10.5); arr(86, 21.5, 89, 17.5); arr(86, 10.5, 89, 15.5)
    ax.text(52, 2.5, 'Evaluated on official split, mixed benign/attack streams, unseen recordings, realistic base rates (50-95% benign).',
            fontsize=6.4, color=INK2)
    save(fig, 'fig01_overview')

# =====================================================================
# Fig 2 - IAT decodes to capture timestamps (v10f)
# =====================================================================
def fig02():
    D = pd.read_csv(R('v10f', 'iat_decoded_per_file.csv')).dropna(subset=['decoded_start_utc'])
    D['t0'] = pd.to_datetime(D['decoded_start_utc']); D['t1'] = pd.to_datetime(D['decoded_end_utc'])
    order = D.groupby('label')['t0'].min().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(S2, 3.0), gridspec_kw={'width_ratios': [2.3, 1]})
    ax = axes[0]
    for i, lab in enumerate(order):
        g = D[D['label'] == lab]; c = FAM_COL[family6(lab)]
        for _, r in g.iterrows():
            ax.plot([r['t0'], max(r['t1'], r['t0'] + pd.Timedelta(minutes=40))], [i, i], lw=4.5 if r['split'] == 'train' else 2.2,
                    color=c, alpha=1 if r['split'] == 'train' else 0.55, solid_capstyle='butt')
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b')); ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_xlabel('Decoded capture time (UTC, 2023) = IAT x k'); ax.grid(axis='y', alpha=0)
    ax.legend(handles=[Patch(color=c, label=f) for f, c in FAM_COL.items()] +
              [plt.Line2D([], [], lw=4.5, color=INK2, label='train file'), plt.Line2D([], [], lw=2.2, color=INK2, alpha=0.55, label='test file')],
              ncol=4, fontsize=6, loc='lower left', frameon=False, bbox_to_anchor=(0, -0.42))
    ax.set_title('(a) One capture session per class; test file = end of session', loc='left')
    ax2 = axes[1]
    k = D['k_window'].astype(int); raw = D['median_huge_IAT']
    for kk, c in [(20, BLUE), (10, ORANGE)]:
        m = k == kk; ax2.scatter(raw[m] / 1e6, (raw[m] * kk) / 1e9, s=10, color=c, label=f'k = {kk} (window size)', zorder=3)
    ax2.set_xlabel('Raw IAT value (x 10$^6$)'); ax2.set_ylabel('IAT x k (x 10$^9$ = Unix time)')
    ax2.axhspan(1.6932, 1.6948, color='#f0efec', zorder=0); ax2.text(raw.min() / 1e6, 1.6950, 'Aug-Sep 2023', fontsize=6, color=INK2)
    ax2.legend(frameon=False, loc='center right'); ax2.set_title('(b) Two raw levels, one date range', loc='left')
    plt.tight_layout(); save(fig, 'fig02_iat_timestamp_decode')

# =====================================================================
# Fig 3 - single-feature scan (v10e P1)  +  Table 1 audit
# =====================================================================
def fig03():
    P = pd.read_csv(R('v10e', 'P1_single_feature_scan.csv')).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(S1, 2.9))
    ax.barh(P['feature'], P['accuracy_%'], color=[ORANGE if f == 'IAT' else BLUE for f in P['feature']], height=0.7)
    for y, v in enumerate(P['accuracy_%']): ax.text(v + 1, y, f'{v:.1f}', va='center', fontsize=6)
    ax.set_xlim(0, 108); ax.set_xlabel('19-class accuracy with ONE feature (%)'); ax.grid(axis='y', alpha=0)
    ax.set_title('Train on official train set, test on official test set', loc='left', fontsize=7)
    save(fig, 'fig03_single_feature_scan')

def tab01():
    # frozen numbers: v8 Table A (3 seeds), v8b CNN (best-val epoch), v10e P1/P2
    p2 = pd.read_csv(R('v10e', 'P2_ablation.csv')).set_index('features')
    rows = [('XGBoost', 'all 42 features', 99.34, 89.46), ('XGBoost', 'IAT only (1 feature)', 98.90, 81.07),
            ('XGBoost', '42 minus IAT', p2.loc['42 minus IAT (41)', 'accuracy_%'], p2.loc['42 minus IAT (41)', 'macro_f1_%']),
            ('XGBoost', '42 minus Number, Weight', p2.loc['42 minus Number+Weight (40)', 'accuracy_%'], p2.loc['42 minus Number+Weight (40)', 'macro_f1_%']),
            ('XGBoost', 'shortcut-free (39)', 71.96, 69.31),
            ('Published CNN [ref]', 'all 45 features', 98.48, 73.57), ('Published CNN [ref]', 'shortcut-free (39)', 77.63, 58.71)]
    table(pd.DataFrame(rows, columns=['Model', 'Features', 'Accuracy (%)', 'Macro-F1 (%)']), 'tab01_shortcut_audit',
          'Shortcut audit on the official CICIoMT2024 test set (19 classes). One feature, IAT, reproduces near-perfect accuracy.', 'tab:audit')

# =====================================================================
# Fig 4 - accuracy vs macro-F1 under artifact removal (frozen v8/v8b numbers)
# =====================================================================
def fig04():
    lab = ['XGB\nall 42', 'XGB\nIAT only', 'XGB\nshortcut-free', 'CNN [ref]\nall 45', 'CNN [ref]\nshortcut-free']
    acc = [99.34, 98.90, 71.96, 98.48, 77.63]; mf1 = [89.46, 81.07, 69.31, 73.57, 58.71]
    x = np.arange(len(lab)); w = 0.38
    fig, ax = plt.subplots(figsize=(S1, 2.2))
    ax.bar(x - w / 2, acc, w, color=BLUE, label='Accuracy'); ax.bar(x + w / 2, mf1, w, color=ORANGE, label='Macro-F1')
    for i in range(len(x)):
        ax.text(x[i] - w / 2, acc[i] + 1, f'{acc[i]:.1f}', ha='center', fontsize=5.5); ax.text(x[i] + w / 2, mf1[i] + 1, f'{mf1[i]:.1f}', ha='center', fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6); ax.set_ylim(0, 108); ax.set_ylabel('%'); ax.grid(axis='x', alpha=0)
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    save(fig, 'fig04_accuracy_vs_macroF1')

# =====================================================================
# Fig 5 - SHAP all vs shortcut-free (v10b)
# =====================================================================
def fig05():
    T = pd.read_csv(R('v10b', 'TABLE_X1_shap_global.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(S2, 2.6))
    for ax, s, ttl in zip(axes, ['All 42 features', 'Shortcut-free 39'], ['(a) With all features', '(b) Shortcut-free']):
        t = T[T['setting'] == s].sort_values('rank').head(12).iloc[::-1]
        ax.barh(t['feature'], t['mean'], xerr=t['std'], color=[ORANGE if f in ('IAT', 'Number', 'Weight') else BLUE for f in t['feature']],
                height=0.7, error_kw={'elinewidth': 0.6, 'ecolor': INK2})
        for y, (v, sd) in enumerate(zip(t['mean'], t['std'].fillna(0))): ax.text(v + sd + 0.3, y, f'{v:.1f}', va='center', fontsize=6)
        ax.set_xlabel('Share of mean |SHAP| (%), 3 seeds'); ax.set_title(ttl, loc='left'); ax.grid(axis='y', alpha=0)
    fig.legend(handles=[Patch(color=ORANGE, label='Recording artifact (IAT / Number / Weight)'), Patch(color=BLUE, label='Behavioural feature')],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1]); save(fig, 'fig05_shap_comparison')

# =====================================================================
# Fig 6 - metric hierarchy (v10a)  +  Fig 7 confusion matrix (v10a npz)
# =====================================================================
LEVELS = ['Binary (attack vs benign)', '6 families (official)', '5 families (DoS+DDoS merged)', '13 types (DoS+DDoS merged)', '19 classes (original)']
def fig06():
    H = pd.read_csv(R('v10a', 'P1_metric_hierarchy_raw.csv'))
    H['grp'] = np.where(H['setting'].str.startswith('static'), 'Static (official test)', 'Mixed stream (mean over L)')
    T = H.groupby(['level', 'grp'])['macro_f1'].mean().mul(100).unstack().reindex(LEVELS)
    A = H.groupby(['level', 'grp'])['accuracy'].mean().mul(100).unstack().reindex(LEVELS)
    fig, ax = plt.subplots(figsize=(S1, 2.3)); y = np.arange(len(LEVELS))
    for j, (g, c) in enumerate([('Static (official test)', BLUE), ('Mixed stream (mean over L)', ORANGE)]):
        ax.barh(y + (j - 0.5) * 0.38, T[g], 0.36, color=c, label=g + ', macro-F1')
        for k_, v in enumerate(T[g]): ax.text(v + 0.8, y[k_] + (j - 0.5) * 0.38, f'{v:.1f}', va='center', fontsize=5.8)
    ax.scatter(A['Static (official test)'], y - 0.19, marker='|', s=60, color='k', zorder=4, label='Static accuracy')
    ax.set_yticks(y); ax.set_yticklabels([l.replace(' (', '\n(') for l in LEVELS], fontsize=6.2); ax.invert_yaxis()
    ax.set_xlim(0, 108); ax.set_xlabel('%'); ax.grid(axis='y', alpha=0); ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.4, -0.18), ncol=2)
    save(fig, 'fig06_metric_hierarchy')
    T2 = pd.concat({'Static acc': A['Static (official test)'], 'Static mF1': T['Static (official test)'],
                    'Stream acc': A['Mixed stream (mean over L)'], 'Stream mF1': T['Mixed stream (mean over L)']}, axis=1).round(2)
    table(T2.reset_index().rename(columns={'level': 'Label granularity'}), 'tab03_metric_hierarchy',
          'Shortcut-free performance at different label granularities. Merging DoS and DDoS of the same protocol recovers 99.3\\% accuracy.', 'tab:hierarchy')

def fig07():
    z = np.load(R('v10a', 'probs_seed42.npz'), allow_pickle=True)
    lab = list(z['labels']); y, p = z['y_static'], z['P_static'].argmax(1)
    order = sorted(range(len(lab)), key=lambda i: (['Benign', 'DDoS', 'DoS', 'MQTT', 'Recon', 'Spoofing'].index(family6(lab[i])), lab[i].replace('DDoS', 'DoS')))
    C = pd.crosstab(pd.Series(y), pd.Series(p)).reindex(index=range(len(lab)), columns=range(len(lab)), fill_value=0).values.astype(float)
    C = C / C.sum(1, keepdims=True).clip(min=1) * 100; C = C[np.ix_(order, order)]; names = [lab[i] for i in order]
    fig, ax = plt.subplots(figsize=(S2 * 0.62, S2 * 0.58))
    im = ax.imshow(C, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#6da7ec', '#1c5cab', '#0d366b']), vmin=0, vmax=100)
    for i in range(len(names)):
        for j in range(len(names)):
            if C[i, j] >= 5: ax.text(j, i, f'{C[i, j]:.0f}', ha='center', va='center', fontsize=4.8, color='white' if C[i, j] > 55 else 'black')
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=5.2); ax.set_yticklabels(names, fontsize=5.2)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.grid(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label('Row %', fontsize=6); cb.ax.tick_params(labelsize=5.5)
    save(fig, 'fig07_confusion_shortcut_free')

# =====================================================================
# Fig 8 - mixed stream, 5 seeds (v8)  +  Fig 9 temporal baselines (v10e P3)
# =====================================================================
def fig08():
    B = pd.read_csv(R('v8', 'B_mixed_raw.csv'))
    g = B.groupby(['model', 'L'])['macro_f1'].agg(['mean', 'std']).mul(100).reset_index()
    fig, ax = plt.subplots(figsize=(S1, 2.2))
    for m, c, mk in [('(a) per-window XGB', BLUE, 'o'), ('(c) context XGB', AQUA, 's'), ('(e) ensemble a+c', ORANGE, 'D')]:
        t = g[g['model'] == m]; ax.errorbar(t['L'], t['mean'], yerr=t['std'], color=c, marker=mk, ms=3.5, lw=1.4, capsize=2, label=m)
    ax.set_xscale('log'); ax.set_xticks(sorted(g['L'].unique())); ax.set_xticklabels(sorted(g['L'].unique()))
    ax.set_xlabel('Segment length L (windows)'); ax.set_ylabel('Macro-F1 (%), 5 seeds'); ax.legend(frameon=False)
    save(fig, 'fig08_mixed_stream_5seeds')
    T = g.pivot(index='model', columns='L', values='mean').round(2)
    T['mean over L'] = B.groupby(['model', 'seed'])['macro_f1'].mean().groupby('model').mean().mul(100).round(2)
    table(T.reset_index(), 'tab04_mixed_stream', 'Mixed-stream macro-F1 (\\%, mean of 5 seeds) by segment length $L$.', 'tab:stream')

def fig09():
    P = pd.read_csv(R('v10e', 'P3_temporal_baseline.csv')); P = P[P['model'] == 'Full'] if 'Full' in set(P['model']) else P[P['model'] == P['model'].iloc[0]]
    fig, ax = plt.subplots(figsize=(S1, 2.2))
    sty = {'(a) per-window': (BLUE, 'o', '-'), '(a) + rolling-avg probs K=5': (GREY, 'v', '--'), '(a) + rolling-avg probs K=10': (GREY, '^', ':'),
           '(e) context ensemble (ours)': (ORANGE, 'D', '-'), '(e) + rolling-avg probs K=5': (YELLOW, 's', '--')}
    for m, (c, mk, ls) in sty.items():
        t = P[P['method'] == m].sort_values('L')
        if len(t): ax.plot(t['L'], t['macro_f1_%'], color=c, marker=mk, ls=ls, ms=3.5, lw=1.3, label=m)
    ax.set_xscale('log'); Ls = sorted(P['L'].unique()); ax.set_xticks(Ls); ax.set_xticklabels(Ls)
    ax.set_xlabel('Segment length L (windows)'); ax.set_ylabel('Macro-F1 (%)'); ax.legend(frameon=False, fontsize=5.8)
    save(fig, 'fig09_temporal_baselines')

# =====================================================================
# Fig 10 - operating point at 95% benign (v10d sweep + v10e validated point)  +  Fig 11 base rate (v10c)
# =====================================================================
def fig10():
    S = pd.read_csv(R('v10d', 'P1_threshold_sweep.csv')); V = pd.read_csv(R('v10e', 'P4_validated_threshold.csv'))
    fig, ax = plt.subplots(figsize=(S1, 2.4))
    for mdl, c in [('Lite-B', BLUE), ('Full', ORANGE)]:
        g = S[(S['model'] == mdl) & (S['benign_share'] == 0.95)].groupby('threshold', sort=False)[['detected_%', 'event_precision_%']].mean()
        if not len(g): continue
        ax.plot(g['detected_%'], g['event_precision_%'], marker='o', ms=3, lw=1.3, color=c, label=f'{mdl}: threshold sweep')
        v = V[(V['model'] == mdl) & (V['benign_share'] == 0.95) & (V['rule'] == 'global t')][['detected_%', 'event_precision_%']].mean()
        a = V[(V['model'] == mdl) & (V['benign_share'] == 0.95) & (V['rule'] == 'argmax (no threshold)')][['detected_%', 'event_precision_%']].mean()
        ax.scatter(v['detected_%'], v['event_precision_%'], marker='*', s=90, color=c, edgecolor='k', lw=0.5, zorder=5,
                   label=f'{mdl}: threshold chosen on validation')
        ax.scatter(a['detected_%'], a['event_precision_%'], marker='x', s=30, color=c, zorder=5)
    ax.scatter([], [], marker='x', color=INK2, label='argmax rule (no threshold)')
    ax.set_xlabel('Attack segments detected (%)'); ax.set_ylabel('Alert (event) precision (%)'); ax.legend(frameon=False, fontsize=5.8, loc='lower left')
    ax.set_title('95% benign traffic, 3-of-5 smoothing', loc='left', fontsize=7)
    save(fig, 'fig10_operating_point')
    keep = V[V['rule'].isin(['argmax (no threshold)', 'global t'])].groupby(['model', 'benign_share', 'rule'], sort=False)[
        ['FA_per_1000_benign', 'event_precision_%', 'window_precision_%', 'detected_%']].mean().round(2).reset_index()
    table(keep, 'tab05_alert_system', 'Alert-level results on test streams; threshold selected on validation streams only.', 'tab:alerts')

def fig11():
    B = pd.read_csv(R('v10c', 'B_baserate_system.csv')); B = B[B['smoothing'] == '3-of-5']
    g = B.groupby(['model', 'benign_share'])[['alarm_precision_%', 'false_alarms_per_1000_benign']].mean().reset_index()
    fig, ax = plt.subplots(figsize=(S1, 2.0)); x = np.arange(3); w = 0.38
    for j, (mdl, c) in enumerate([('Full', BLUE), ('Lite-B', ORANGE)]):
        t = g[g['model'] == mdl].sort_values('benign_share')
        ax.bar(x + (j - 0.5) * w, t['alarm_precision_%'], w, color=c, label=mdl)
        for i, v in enumerate(t['alarm_precision_%']): ax.text(x[i] + (j - 0.5) * w, v + 1.5, f'{v:.0f}', ha='center', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(['50% benign', '80% benign', '95% benign']); ax.set_ylim(0, 108)
    ax.set_ylabel('Window alarm precision (%)'); ax.legend(frameon=False, ncol=2); ax.grid(axis='x', alpha=0)
    ax.set_title('Argmax rule, 3-of-5 smoothing', loc='left', fontsize=7)
    save(fig, 'fig11_base_rate')

# =====================================================================
# Fig 12 - generalisation: unseen recordings + zero-day (v10e P5, P6)
# =====================================================================
def fig12():
    P5 = pd.read_csv(R('v10e', 'P5_leave_one_recording_out.csv')).groupby(['features', 'split'], sort=False)[['acc_19_%', 'acc_13type_%']].mean()
    P6 = pd.read_csv(R('v10e', 'P6_zero_day.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(S2, 2.2), gridspec_kw={'width_ratios': [1, 1.4]})
    ax = axes[0]; cats = [('IAT only', 'acc_19_%', 'IAT only'), ('Shortcut-free 39', 'acc_19_%', 'Shortcut-free\n(19 classes)'),
                          ('Shortcut-free 39', 'acc_13type_%', 'Shortcut-free\n(13 types)')]
    x = np.arange(len(cats)); w = 0.38
    for j, (sp, c) in enumerate([('same recordings (random split)', GREY), ('unseen recording (LORO)', BLUE)]):
        vals = [P5.loc[(f, sp), col] for f, col, _ in cats]
        ax.bar(x + (j - 0.5) * w, vals, w, color=c, label=sp)
        for i, v in enumerate(vals): ax.text(x[i] + (j - 0.5) * w, v + 1, f'{v:.1f}', ha='center', fontsize=5.6)
    ax.set_xticks(x); ax.set_xticklabels([c[2] for c in cats], fontsize=6); ax.set_ylim(0, 112); ax.set_ylabel('Accuracy (%)')
    ax.legend(frameon=False, fontsize=5.8, loc='upper left', ncol=1); ax.grid(axis='x', alpha=0); ax.set_ylim(0, 135); ax.set_title('(a) DoS/DDoS classes, held-out capture', loc='left')
    ax = axes[1]; fam = ['DDoS', 'DoS', 'MQTT', 'Recon', 'Spoofing']; x = np.arange(len(fam))
    for j, (f, c) in enumerate([('All 42 (with artifacts)', ORANGE), ('Shortcut-free 39', BLUE)]):
        t = P6[P6['features'] == f].set_index('held_out_family').reindex(fam)
        ax.bar(x + (j - 0.5) * w, t['recall_on_unseen_family_%'], w, color=c, label=f)
        for i, v in enumerate(t['recall_on_unseen_family_%']): ax.text(x[i] + (j - 0.5) * w, v + 1, f'{v:.0f}', ha='center', fontsize=5.6)
    ax.set_xticks(x); ax.set_xticklabels(fam); ax.set_ylim(0, 112); ax.set_ylabel('Recall on unseen family (%)')
    ax.legend(frameon=False, fontsize=5.8, loc='upper left', ncol=2); ax.set_ylim(0, 125); ax.grid(axis='x', alpha=0); ax.set_title('(b) Zero-day: family never seen in training', loc='left')
    plt.tight_layout(); save(fig, 'fig12_generalisation')

# =====================================================================
# Fig 13 - per-class F1 at 19 vs 13 granularity (v10a)
# =====================================================================
def fig13():
    P = pd.read_csv(R('v10a', 'P1_per_class_19_vs_13.csv'), index_col=0).sort_values('F1_13type')
    fig, ax = plt.subplots(figsize=(S1, 3.0)); y = np.arange(len(P))
    ax.hlines(y, P['F1_19class'], P['F1_13type'], color=GREY, lw=1)
    ax.scatter(P['F1_19class'], y, color=BLUE, s=12, label='19 classes', zorder=3)
    ax.scatter(P['F1_13type'], y, color=ORANGE, s=12, label='13 types (DoS+DDoS merged)', zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(P.index, fontsize=5.8); ax.set_xlim(0, 103); ax.set_xlabel('Per-class F1 (%), shortcut-free, official test')
    ax.axhline(4.5, color=INK2, lw=0.5, ls='--'); ax.text(2, 2.0, 'low-rate attacks:\nstill unsolved', fontsize=6, color=INK2)
    ax.legend(frameon=False, loc='lower right', fontsize=6); ax.grid(axis='y', alpha=0)
    save(fig, 'fig13_per_class_f1')

# =====================================================================
# Fig 14 - Lite vs Full (frozen v9b numbers)  +  Table 6 generalisation summary
# =====================================================================
def fig14():
    V = pd.DataFrame([('Tiny', 4, 80, 71.03, 1.26, 5.39), ('Lite-B (default)', 6, 150, 72.57, 1.98, 21.96), ('Full', 8, 1000, 72.41, 18.24, 209.42)],
                     columns=['variant', 'depth', 'trees', 'macro_f1', 'latency_ms', 'size_mb'])
    fig, ax = plt.subplots(figsize=(S1, 2.0))
    ax.scatter(V['latency_ms'], V['macro_f1'], s=V['size_mb'] * 1.2 + 15, color=[GREY, ORANGE, BLUE], alpha=0.9, edgecolor='k', lw=0.4)
    for _, r in V.iterrows(): ax.annotate(f"{r['variant']}\n{r['size_mb']:.0f} MB", (r['latency_ms'], r['macro_f1']), textcoords='offset points', xytext=(8, -4), fontsize=6)
    ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(['1', '2', '5', '10', '20']); ax.set_xlim(0.9, 40); ax.set_xlabel('Latency per window (ms, 1 CPU thread)'); ax.set_ylabel('Mixed-stream macro-F1 (%)'); ax.set_ylim(70, 73.5)
    save(fig, 'fig14_model_variants')
    table(V, 'tab07_model_variants', 'Model variants (v9b, mixed-stream test, 1 CPU thread). Lite-B is used as default; macro-F1 from v10e P3 / v9b.', 'tab:variants')

def tab06():
    P5 = pd.read_csv(R('v10e', 'P5_leave_one_recording_out.csv')).groupby(['features', 'split'], sort=False)[['acc_19_%', 'acc_13type_%']].mean().round(2).reset_index()
    table(P5, 'tab06_leave_one_recording_out', 'Accuracy on DoS/DDoS classes when the test capture was never seen in training (2 folds).', 'tab:loro')
    P8 = pd.read_csv(R('v10e', 'P8_bootstrap_ci.csv'))
    table(P8, 'tab08_bootstrap_ci', 'Bootstrap 95\\% confidence intervals (window resampling).', 'tab:ci')

for name, fn in [('fig01', fig01), ('fig02', fig02), ('fig03', fig03), ('tab01', tab01), ('fig04', fig04), ('fig05', fig05),
                 ('fig06+tab03', fig06), ('fig07', fig07), ('fig08+tab04', fig08), ('fig09', fig09), ('fig10+tab05', fig10),
                 ('fig11', fig11), ('fig12', fig12), ('fig13', fig13), ('fig14+tab07', fig14), ('tab06+tab08', tab06)]:
    run(name, fn)
print(f'\nDone: {len(DONE)} | Skipped: {len(SKIP)}')
for s in SKIP: print('  -', s)
print('Output folder:', OUT)
