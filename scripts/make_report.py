"""Full evaluation report (HTML + PNG + JSON) for the active IDS model, plus the 12-model comparison.

Usage:  python scripts/make_report.py                 # default model (config.yaml)
        python scripts/make_report.py --model full
Output: reports/<model>/report.html, reports/<model>/figures/*.png, reports/<model>/metrics.json
"""
import argparse
import base64
import datetime as dt
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd   # noqa: E402
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix, f1_score,   # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from iomt_ids import IDSPipeline, alert_metrics, load_config   # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'
GREY, INK2 = '#9a9890', '#52514e'
FAM_ORDER = ['Benign', 'DDoS', 'DoS', 'MQTT', 'Recon', 'Spoofing']
FAM_COL = dict(zip(FAM_ORDER, [GREY, BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#ffffff', '#cde2fb', '#6da7ec', '#1c5cab', '#0d366b'])
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.25, 'axes.axisbelow': True, 'savefig.bbox': 'tight', 'figure.dpi': 110})


def family(l):
    if l == 'Benign': return 'Benign'
    if l.startswith('MQTT'): return 'MQTT'
    if l.startswith('Recon'): return 'Recon'
    if 'Spoof' in l: return 'Spoofing'
    return 'DDoS' if l.startswith('DDoS') else 'DoS'


class Report:
    def __init__(self, out):
        self.out = out; (out / 'figures').mkdir(parents=True, exist_ok=True); self.parts = []

    def fig(self, fig, name, caption):
        p = self.out / 'figures' / f'{name}.png'; fig.savefig(p, dpi=150); plt.close(fig)
        b64 = base64.b64encode(p.read_bytes()).decode()
        self.parts.append(f'<figure><img src="data:image/png;base64,{b64}"><figcaption>{caption}</figcaption></figure>')

    def h(self, text, level=2): self.parts.append(f'<h{level}>{text}</h{level}>')
    def p(self, text): self.parts.append(f'<p>{text}</p>')
    def table(self, df, index=False): self.parts.append(df.to_html(index=index, float_format=lambda v: f'{v:.2f}', border=0, classes='t'))

    def save(self, title):
        css = ('body{font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#0b0b0b}'
               'h1{font-size:26px}h2{border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:36px}'
               'figure{margin:16px 0}img{max-width:100%}figcaption{color:#52514e;font-size:13px}'
               '.t{border-collapse:collapse;font-size:13px;margin:8px 0}.t td,.t th{padding:4px 10px;border-bottom:1px solid #eee;text-align:right}'
               '.t th{background:#f3f2ef}.t td:first-child,.t th:first-child{text-align:left}.note{color:#52514e;font-size:13px}')
        html = f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head><body>' + ''.join(self.parts) + '</body></html>'
        (self.out / 'report.html').write_text(html, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default=None); ap.add_argument('--latency-windows', type=int, default=1500)
    a = ap.parse_args()
    cfg = load_config(ROOT / 'config.yaml', a.model); model = cfg['active_model']
    out = ROOT / 'reports' / model; R = Report(out); M = {}
    print(f'Building report for model "{model}" ...')
    df = pd.read_parquet(cfg['demo_stream']); ids = IDSPipeline(cfg)
    X = ids.guard.transform_frame(df); y = df['label'].to_numpy(); labels = cfg['labels']; L = np.array(labels)
    t0 = time.time(); P = ids.detector.predict_proba_batch(X, cfg['context_k']); batch_s = time.time() - t0
    pred = L[P.argmax(1)]; present = [l for l in labels if l in set(y)]
    bi = labels.index('Benign'); score = 1 - P[:, bi]; is_att = y != 'Benign'

    R.h(f'IoMT-IDS evaluation report — model "{model}"', 1)
    R.p(f'Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Data: demo stream ({len(y):,} windows of the official CICIoMT2024 '
        f'test set, mixed with benign; {(~is_att).mean()*100:.0f}% benign). Features: {len(cfg["features"])} leakage-free '
        f'(IAT, Number, Weight and 3 redundant columns removed). Context K = {cfg["context_k"]}, fusion w = {cfg["fusion_w"]}.')

    # ---------------- 1. headline metrics
    yi = np.array([labels.index(v) for v in y])
    M['accuracy'] = accuracy_score(y, pred) * 100
    M['macro_f1'] = f1_score(y, pred, labels=present, average='macro') * 100
    M['weighted_f1'] = f1_score(y, pred, labels=present, average='weighted') * 100
    try:
        M['roc_auc_macro_ovr'] = roc_auc_score(yi, P[:, [labels.index(l) for l in present]] / P[:, [labels.index(l) for l in present]].sum(1, keepdims=True),
                                               multi_class='ovr', average='macro', labels=[labels.index(l) for l in present]) * 100
    except Exception:
        M['roc_auc_macro_ovr'] = None
    M['binary_roc_auc'] = roc_auc_score(is_att, score) * 100
    thr = cfg.get('attack_threshold')
    flag_arg, flag_thr = P.argmax(1) != bi, (score >= thr) if thr is not None else P.argmax(1) != bi
    for name, fl in [('argmax', flag_arg), (f'threshold_{thr}', flag_thr)]:
        M[f'binary_{name}'] = {'attack_recall_%': float(fl[is_att].mean() * 100), 'benign_FPR_%': float(fl[~is_att].mean() * 100),
                               'precision_%': float(is_att[fl].mean() * 100) if fl.any() else None}
        if 'segment' in df:
            M[f'alerts_{name}'] = alert_metrics(y, df['segment'].to_numpy(), fl, cfg['smoothing_m'], cfg['smoothing_n'])
    head = pd.DataFrame([('Accuracy (19 classes)', M['accuracy']), ('Macro-F1', M['macro_f1']), ('Weighted-F1', M['weighted_f1']),
                         ('ROC-AUC macro (one-vs-rest)', M['roc_auc_macro_ovr'] or np.nan), ('Attack vs benign ROC-AUC', M['binary_roc_auc'])],
                        columns=['Metric', '%'])
    R.h('1. Headline metrics'); R.table(head)
    rows = [(k.replace('binary_', ''), v['attack_recall_%'], v['benign_FPR_%'], v['precision_%']) for k, v in M.items() if k.startswith('binary_') and isinstance(v, dict)]
    R.p('<b>Attack vs benign (per window)</b>'); R.table(pd.DataFrame(rows, columns=['Rule', 'Attack recall %', 'Benign FPR %', 'Precision %']))
    if 'segment' in df:
        al = pd.DataFrame([{'Rule': k.replace('alerts_', ''), **v} for k, v in M.items() if k.startswith('alerts_')])
        R.p(f"<b>Alert level ({cfg['smoothing_m']}-of-{cfg['smoothing_n']} smoothing)</b>"); R.table(al)

    # ---------------- 2. per-class report + confusion
    rep = pd.DataFrame(classification_report(y, pred, labels=present, output_dict=True, zero_division=0)).T.loc[present]
    rep[['precision', 'recall', 'f1-score']] *= 100; rep['support'] = rep['support'].astype(int)
    M['per_class'] = rep[['precision', 'recall', 'f1-score', 'support']].round(2).to_dict('index')
    R.h('2. Per-class results'); R.table(rep[['precision', 'recall', 'f1-score', 'support']].rename_axis('class').reset_index())
    order = sorted(present, key=lambda l: (FAM_ORDER.index(family(l)), l.replace('DDoS', 'DoS')))
    fig, ax = plt.subplots(figsize=(8, 4.2)); x = np.arange(len(order)); w = 0.27
    for j, (col, c) in enumerate([('precision', BLUE), ('recall', ORANGE), ('f1-score', AQUA)]):
        ax.bar(x + (j - 1) * w, rep.loc[order, col], w, color=c, label=col)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=60, ha='right', fontsize=7); ax.set_ylabel('%'); ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0)); ax.grid(axis='x', alpha=0)
    R.fig(fig, 'per_class_prf', 'Per-class precision, recall and F1 (grouped by attack family).')
    C = confusion_matrix(y, pred, labels=order).astype(float); Cn = C / C.sum(1, keepdims=True).clip(min=1) * 100
    fig, ax = plt.subplots(figsize=(7.5, 6.6)); im = ax.imshow(Cn, cmap=SEQ, vmin=0, vmax=100)
    for i in range(len(order)):
        for j in range(len(order)):
            if Cn[i, j] >= 5: ax.text(j, i, f'{Cn[i, j]:.0f}', ha='center', va='center', fontsize=6, color='white' if Cn[i, j] > 55 else 'black')
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order))); ax.set_xticklabels(order, rotation=60, ha='right', fontsize=6.5)
    ax.set_yticklabels(order, fontsize=6.5); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.grid(False); plt.colorbar(im, fraction=0.04, label='row %')
    R.fig(fig, 'confusion_matrix', 'Confusion matrix (row-normalised %). The 2x2 blocks are DoS vs DDoS of the same protocol.')
    fam_t, fam_p = pd.Series(y).map(family), pd.Series(pred).map(family)
    CF = pd.crosstab(fam_t, fam_p, normalize='index').reindex(index=FAM_ORDER, columns=FAM_ORDER).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(4.6, 3.8)); im = ax.imshow(CF.values, cmap=SEQ, vmin=0, vmax=100)
    for i in range(6):
        for j in range(6): ax.text(j, i, f'{CF.values[i, j]:.0f}', ha='center', va='center', fontsize=7, color='white' if CF.values[i, j] > 55 else 'black')
    ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels(FAM_ORDER, rotation=30); ax.set_yticklabels(FAM_ORDER); ax.grid(False)
    ax.set_xlabel('Predicted family'); ax.set_ylabel('True family')
    R.fig(fig, 'confusion_family', 'Confusion by attack family (row %).')

    # ---------------- 3. ROC / PR
    R.h('3. ROC and precision-recall curves')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    fpr, tpr, _ = roc_curve(is_att, score); axes[0].plot(fpr, tpr, color=INK2, lw=2, label=f'attack vs benign (AUC {auc(fpr, tpr):.3f})')
    pre, rec, _ = precision_recall_curve(is_att, score); axes[1].plot(rec, pre, color=INK2, lw=2, label='attack vs benign')
    fams = np.array([family(l) for l in labels])
    for f in FAM_ORDER[1:]:
        tgt = fam_t.to_numpy() == f
        if tgt.sum() == 0 or tgt.all(): continue
        s = P[:, fams == f].sum(1); fpr, tpr, _ = roc_curve(tgt, s); pre, rec, _ = precision_recall_curve(tgt, s)
        axes[0].plot(fpr, tpr, color=FAM_COL[f], lw=1.3, label=f'{f} (AUC {auc(fpr, tpr):.3f})'); axes[1].plot(rec, pre, color=FAM_COL[f], lw=1.3, label=f)
    axes[0].plot([0, 1], [0, 1], ls='--', color=GREY, lw=0.8); axes[0].set_xlabel('False positive rate'); axes[0].set_ylabel('True positive rate')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[0].legend(frameon=False, fontsize=7); axes[1].legend(frameon=False, fontsize=7)
    plt.tight_layout(); R.fig(fig, 'roc_pr', 'Left: ROC curves (one-vs-rest by attack family, and attack vs benign). Right: precision-recall curves.')

    # ---------------- 4. calibration + score distribution + threshold trade-off
    R.h('4. Confidence, calibration and alert threshold')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(score, bins) - 1, 0, 9)
    pts = [(score[idx == i].mean(), is_att[idx == i].mean()) for i in range(10) if (idx == i).sum() >= 20]
    axes[0].plot([0, 1], [0, 1], ls='--', color=GREY, lw=0.8); axes[0].plot(*zip(*pts), marker='o', color=BLUE, lw=2)
    ece = sum((idx == i).mean() * abs(score[idx == i].mean() - is_att[idx == i].mean()) for i in range(10) if (idx == i).any()) * 100
    M['ece_attack_score_%'] = float(ece)
    axes[0].set_xlabel('Predicted P(attack)'); axes[0].set_ylabel('Observed attack fraction'); axes[0].set_title(f'Reliability (ECE {ece:.2f}%)', fontsize=9)
    axes[1].hist(score[~is_att], bins=40, color=GREY, alpha=0.8, label='benign windows'); axes[1].hist(score[is_att], bins=40, color=ORANGE, alpha=0.7, label='attack windows')
    if thr is not None: axes[1].axvline(thr, color=INK2, ls='--', lw=1); axes[1].text(thr, axes[1].get_ylim()[1] * 0.9, f' t={thr}', fontsize=8)
    axes[1].set_yscale('log'); axes[1].set_xlabel('P(attack)'); axes[1].legend(frameon=False, fontsize=7)
    plt.tight_layout(); R.fig(fig, 'calibration', 'Left: calibration of the attack score. Right: score distribution of benign vs attack windows (log scale).')
    if 'segment' in df:
        ts = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
        tr = pd.DataFrame([{'threshold': t, **alert_metrics(y, df['segment'].to_numpy(), score >= t, cfg['smoothing_m'], cfg['smoothing_n'])} for t in ts])
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.plot(tr['threshold'], tr['event_precision_%'], marker='o', color=BLUE, label='alert precision %')
        ax.plot(tr['threshold'], tr['attack_segments_detected_%'], marker='s', color=ORANGE, label='attack segments detected %')
        ax2 = ax.twinx(); ax2.plot(tr['threshold'], tr['false_alarms_per_1000_benign'], marker='^', color=GREY, ls='--', label='false alarms / 1000 benign')
        ax2.set_ylabel('false alarms / 1000 benign', color=INK2); ax2.spines['right'].set_visible(True); ax2.grid(False)
        if thr is not None: ax.axvline(thr, color=INK2, ls=':', lw=1)
        ax.set_xlabel('P(attack) threshold'); ax.set_ylabel('%'); h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
        R.fig(fig, 'threshold_tradeoff', 'Alert-level trade-off on the demo stream. The configured threshold is marked (chosen on validation data in the research).')
        M['threshold_sweep'] = tr.to_dict('records')

    # ---------------- 5. latency
    R.h('5. Speed (1 CPU thread, full pipeline incl. Python overhead)')
    recs = df.drop(columns=[c for c in ('label', 'segment') if c in df]).head(a.latency_windows).to_dict('records'); ids.reset()
    lat = np.array([ids.process(i, r).latency_ms for i, r in enumerate(recs)])[50:]
    M['latency_ms'] = {'median': float(np.median(lat)), 'p95': float(np.percentile(lat, 95)), 'throughput_windows_per_s': float(1000 / lat.mean())}
    M['batch_windows_per_s'] = float(len(X) / batch_s)
    M['model_size_MB'] = float(sum((Path(cfg['models_dir']) / f).stat().st_size for f in ('m_window.json', 'm_context.json')) / 1e6)
    fig, ax = plt.subplots(figsize=(6, 3)); ax.hist(lat, bins=50, color=BLUE)
    ax.axvline(np.median(lat), color=INK2, ls='--'); ax.set_xlabel('Latency per window (ms)'); ax.set_ylabel('windows')
    R.fig(fig, 'latency', f"Median {M['latency_ms']['median']:.2f} ms, p95 {M['latency_ms']['p95']:.2f} ms, "
          f"{M['latency_ms']['throughput_windows_per_s']:.0f} windows/s streaming; batch {M['batch_windows_per_s']:.0f} windows/s; model size {M['model_size_MB']:.1f} MB.")

    # ---------------- 6. explainability
    R.h('6. Explainability (TreeSHAP)')
    ex = ids.explainer; ov_w, pc_w = ex.global_importance(X, y, 'window')
    ctxX = np.hstack([X, pd.DataFrame(X).rolling(cfg['context_k'], min_periods=1).mean().to_numpy(),
                      pd.DataFrame(X).rolling(cfg['context_k'], min_periods=1).std().fillna(0).to_numpy()]).astype('float32')
    ov_c, _ = ex.global_importance(ctxX, None, 'context')
    M['shap_top_window'] = (ov_w.head(10) / ov_w.sum() * 100).round(2).to_dict(); M['shap_top_context'] = (ov_c.head(10) / ov_c.sum() * 100).round(2).to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, ov, ttl in [(axes[0], ov_w, 'Per-window model'), (axes[1], ov_c, 'Context model (x, mean10, std10)')]:
        t = (ov.head(15) / ov.sum() * 100).iloc[::-1]
        ax.barh(t.index, t.values, color=[ORANGE if n.startswith(('mean10_', 'std10_')) else BLUE for n in t.index], height=0.7)
        ax.set_title(ttl, fontsize=9); ax.set_xlabel('share of mean |SHAP| (%)'); ax.tick_params(axis='y', labelsize=7); ax.grid(axis='y', alpha=0)
    plt.tight_layout(); R.fig(fig, 'shap_global', 'Global feature importance. Orange = context features (rolling mean/std over the last 10 windows).')
    if pc_w is not None:
        top = ov_w.head(12).index; H = pc_w.loc[[c for c in order if c in pc_w.index], top]; H = H.div(H.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(8, 5.6)); im = ax.imshow(H.values, cmap=SEQ, aspect='auto')
        ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=50, ha='right', fontsize=7); ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index, fontsize=7)
        ax.grid(False); plt.colorbar(im, fraction=0.03, label='% of class SHAP')
        R.fig(fig, 'shap_per_class', 'Which features drive each class (share of |SHAP| towards the true class, top-12 global features).')
    # one example alert explanation (waterfall)
    att_idx = np.flatnonzero(is_att & (pred == y))
    if len(att_idx):
        i = int(att_idx[len(att_idx) // 2]); dfx, base = ex.explain_full(X[i], y[i], model='window'); t = dfx.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh([f"{r.feature} = {r.value:.3g}" for r in t.itertuples()], t['contribution'], color=[BLUE if v > 0 else RED for v in t['contribution']], height=0.7)
        ax.axvline(0, color=INK2, lw=0.8); ax.set_xlabel(f'SHAP contribution to "{y[i]}" (log-odds); base value {base:.2f}'); ax.tick_params(axis='y', labelsize=7)
        R.fig(fig, 'shap_example', f'Example explanation of one correctly detected {y[i]} window (blue pushes towards the class, red against).')

    # ---------------- 7. model comparison (from research notebook v12)
    zoo = ROOT / 'data' / 'zoo'
    if (zoo / 'zoo_all_runs.csv').exists():
        R.h('7. Model comparison — 12 models (research notebook v12, leakage-free features)')
        Z = pd.read_csv(zoo / 'zoo_all_runs.csv'); S = Z[Z['protocol'] == 'S static'].sort_values('macro_f1')
        Mm = pd.read_csv(zoo / 'zoo_stream_mean.csv').sort_values('macro_f1')
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].barh(S['model'], S['macro_f1'], color=BLUE, height=0.6, label='macro-F1'); axes[0].scatter(S['accuracy'], S['model'], color=ORANGE, zorder=3, s=18, label='accuracy')
        axes[0].set_title('Official test (one window at a time)', fontsize=9); axes[0].set_xlim(0, 100); axes[0].legend(frameon=False, fontsize=7); axes[0].tick_params(axis='y', labelsize=7)
        lab = [f"{m} [{v}]" for m, v in zip(Mm['model'], Mm['view'])]
        axes[1].barh(lab, Mm['macro_f1'], color=[ORANGE if 'ours' in m else (AQUA if 'sequence' in v else BLUE) for m, v in zip(Mm['model'], Mm['view'])], height=0.6)
        axes[1].set_title('Mixed hospital-like stream (mean of L=10, 100)', fontsize=9); axes[1].set_xlim(0, 100); axes[1].tick_params(axis='y', labelsize=6.5)
        for ax in axes: ax.set_xlabel('%'); ax.grid(axis='y', alpha=0)
        plt.tight_layout(); R.fig(fig, 'model_comparison', 'Macro-F1 of all models. Orange = our context ensemble, aqua = sequence deep-learning models.')
        R.table(Mm.sort_values('macro_f1', ascending=False)[['model', 'view', 'accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall', 'roc_auc_macro_ovr']])
        R.p('<span class="note">Single training seed. Training/inference times were measured on Colab (XGBoost on GPU, LightGBM on CPU) and are not comparable; see section 5 for laptop latency.</span>')
        if (zoo / 'zoo_per_class_f1.csv').exists():
            PCF = pd.read_csv(zoo / 'zoo_per_class_f1.csv', index_col=0)
            keep = [k for k in PCF.index if k.endswith('S static | window')]
            H = PCF.loc[keep, [c for c in order if c in PCF.columns]]; H.index = [k.split(' | ')[0] for k in H.index]
            fig, ax = plt.subplots(figsize=(10, 3.8)); im = ax.imshow(H.values, cmap=SEQ, vmin=0, vmax=100, aspect='auto')
            ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, rotation=60, ha='right', fontsize=6.5); ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index, fontsize=7)
            ax.grid(False); plt.colorbar(im, fraction=0.025, label='F1 %')
            R.fig(fig, 'model_per_class', 'Per-class F1 of every model on the official test set: all models fail on the same low-rate classes (ARP, VulScan).')

    R.h('8. Notes and limitations')
    R.p('Trace-driven replay of the CICIoMT2024 testbed (no live network). IAT is excluded because it encodes the capture timestamp '
        '(IAT x window size = Unix time). DoS vs DDoS of the same protocol is weakly identifiable from per-window features; '
        'low-rate attacks (ARP spoofing, Recon-VulScan) remain the hardest classes.')
    R.save(f'IoMT-IDS report — {model}')
    (out / 'metrics.json').write_text(json.dumps(M, indent=1, default=float))
    print(f"Accuracy {M['accuracy']:.2f}% | macro-F1 {M['macro_f1']:.2f}% | median latency {M['latency_ms']['median']:.2f} ms")
    print('Report:', out / 'report.html')


if __name__ == '__main__':
    main()
