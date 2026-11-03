"""IoMT-IDS dashboard — live trace replay of CICIoMT2024 traffic.

Run:  streamlit run app/dashboard.py
"""
import re
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from iomt_ids import IDSPipeline, available_models, load_config   # noqa: E402

# ---------------------------------------------------------------- constants
BLUE, ORANGE = '#2a78d6', '#eb6834'                      # categorical slots 1-2
CRITICAL, SERIOUS, GOOD = '#d03b3b', '#ec835a', '#0ca30c'  # status palette
BAND = '#9a9890'
SEV_COLOR = {'HIGH': CRITICAL, 'MEDIUM': SERIOUS}
SEV_ICON = {'HIGH': '🔴', 'MEDIUM': '🟠'}

# Paper results (research log, v8/v8b/v9b). Shown in the "Why leakage-free?" tab.
AUDIT = pd.DataFrame([
    ('XGBoost · all 42 features', 99.34, 89.46),
    ('XGBoost · IAT ONLY (1 feature)', 98.90, 81.07),
    ('XGBoost · best other single feature (Min)', 63.83, 33.14),
    ('XGBoost · leakage-free (39)', 71.96, 69.31),
    ('Published CNN · all 45 features', 98.48, 73.57),
    ('Published CNN · leakage-free', 77.63, 58.71),
], columns=['setting', 'Accuracy', 'Macro-F1'])
VARIANTS = pd.DataFrame([
    ('lite (default)', '6 / 150', 0.7257, 1.98, 460, 22.0),
    ('full', '8 / 1000', 0.7241, 18.24, 50, 209.4),
    ('tiny', '4 / 80', 0.7103, 1.26, 762, 5.4),
], columns=['Model', 'Depth / trees', 'Macro-F1', 'Latency ms', 'Windows/s', 'Size MB'])


def family(label):
    if label == 'Benign': return 'Benign'
    if label == 'ARP_Spoofing': return 'Spoofing'
    return label.split('-')[0]


# ---------------------------------------------------------------- page
st.set_page_config(page_title='IoMT-IDS', page_icon='🛡️', layout='wide')
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: 10px 14px;}
.chip {display:inline-block; padding:2px 10px; margin:0 6px 6px 0; border-radius:999px;
       border:1px solid rgba(128,128,128,.35); font-size:.82rem;}
.alert-card {border-left: 6px solid var(--sev); border-radius: 8px; padding: 10px 14px;
             background: rgba(128,128,128,.07);}
</style>""", unsafe_allow_html=True)

st.title('🛡️ IoMT-IDS — Leakage-Aware, Context-Aware Intrusion Detection')
st.markdown(
    '<span class="chip">CICIoMT2024 · 19 classes</span>'
    '<span class="chip">Artifact guard: IAT (= capture timestamp) removed</span>'
    '<span class="chip">Context buffer K = 10</span>'
    '<span class="chip">XGBoost ensemble</span>'
    '<span class="chip">Validated alert threshold + m-of-n smoothing</span>'
    '<span class="chip">Per-alert SHAP reasons</span>', unsafe_allow_html=True)
st.caption('Trace-driven replay of real testbed traffic (CICIoMT2024 test set, mixed with benign). '
           'Not connected to a live network.')

models = available_models(ROOT / 'config.yaml')
if not models:
    st.error('No model found. Put the 3 model files from Colab into models/lite/ (or models/).'); st.stop()

# ---------------------------------------------------------------- sidebar
ss = st.session_state
with st.sidebar:
    st.header('Controls')
    model = st.selectbox('Model', models, help='lite = default (22 MB, ~2 ms) · full = 209 MB · tiny = 5 MB')
    scenario = st.selectbox('Traffic scenario', ['Mixed hospital traffic', 'Focus: DDoS', 'Focus: DoS',
                                                 'Focus: MQTT', 'Focus: Recon', 'Focus: Spoofing'])
    m = st.slider('Alert when m attack windows…', 1, 10, 3)
    n = st.slider('…out of the last n windows', 1, 10, 5)
    n = max(n, m)
    rule = st.radio('Alert rule', ['Validated threshold', 'Argmax (no threshold)'], horizontal=False,
                    help='Threshold on P(attack) chosen on validation streams (95% benign): ~16x fewer false alarms.')
    _thr0 = load_config(ROOT / 'config.yaml', model).get('attack_threshold') or 0.95
    thr = st.slider('P(attack) threshold', 0.50, 0.99, float(_thr0), 0.01,
                    disabled=rule != 'Validated threshold') if rule == 'Validated threshold' else None
    speed = st.select_slider('Windows per tick (~1 s)', [25, 50, 100, 200, 400], 100)
    c_a, c_b = st.columns(2)
    play = c_a.button('▶ Play', type='primary', width='stretch')
    pause = c_b.button('⏸ Pause', width='stretch')
    step_once = st.button('⏭ Step one tick', width='stretch')
    reset = st.button('⟲ Reset', width='stretch')

cfg = load_config(ROOT / 'config.yaml', model)
key = (model, scenario, m, n, thr)


def build_stream(df, scenario):
    if scenario.startswith('Focus'):
        fam = scenario.split(': ')[1]
        seg_lab = df.groupby('segment', sort=False)['label'].first()
        att = [s for s, l in seg_lab.items() if family(l) == fam]
        ben = [s for s, l in seg_lab.items() if l == 'Benign']
        order = [x for pair in zip(ben, att) for x in pair]
        df = pd.concat([df[df['segment'] == s] for s in order], ignore_index=True)
    return df.reset_index(drop=True)


if reset or 'ids' not in ss or ss.get('key') != key:
    cfg['smoothing_m'], cfg['smoothing_n'] = m, n
    ss.ids = IDSPipeline(cfg); ss.ids.set_threshold(thr)
    df = build_stream(pd.read_parquet(cfg['demo_stream']), scenario)
    ss.labels = df['label'].tolist()
    ss.segs = df['segment'].tolist() if 'segment' in df else [0] * len(df)
    ss.recs = df.drop(columns=[c for c in ('label', 'segment') if c in df]).to_dict('records')
    ss.pos, ss.hist, ss.alerts, ss.key = 0, [], [], key
    ss.playing = ss.get('playing', False) and not reset
if play: ss.playing = True
if pause: ss.playing = False


def advance(k):
    end = min(ss.pos + k, len(ss.recs))
    for i in range(ss.pos, end):
        d = ss.ids.process(i, ss.recs[i])
        active = ss.ids.alerts.active
        ss.hist.append({'window': i, 'segment': ss.segs[i], 'true': ss.labels[i], 'pred': d.predicted,
                        'score': d.attack_score, 'latency': d.latency_ms, 'flagged': d.flagged,
                        'alarm': active is not None, 'alarm_type': active})
        if d.alert:
            a = d.alert
            ss.alerts.append({'window': a.window_index, 'event': a.event, 'type': a.attack_type,
                              'severity': a.severity, 'confidence': a.confidence, 'flagged': a.windows_flagged,
                              'true': ss.labels[i], 'reasons': a.reasons,
                              'x': ss.ids.last_x.tolist(), 'ctx': ss.ids.last_ctx.tolist()})
    ss.pos = end
    if ss.pos >= len(ss.recs): ss.playing = False


if step_once: advance(speed)


# ---------------------------------------------------------------- helpers
def segment_stats(h):
    """Attack segments detected + delay (alarm on with an attack prediction, inside the segment)."""
    det, delay = 0, []
    for seg, g in h.groupby('segment', sort=False):
        if g['true'].iat[0] == 'Benign': continue
        hit = g[(g['alarm']) & (g['flagged'])]
        if len(hit):
            det += 1; delay.append(int(hit['window'].iat[0] - g['window'].iat[0]))
    n_att = sum(1 for _, g in h.groupby('segment', sort=False) if g['true'].iat[0] != 'Benign')
    return n_att, det, (float(np.median(delay)) if delay else None)


def parse_reasons(reasons):
    rows = []
    for r in reasons:
        mm = re.match(r'(.+) \(([+-])([\d.]+)\)', r)
        if mm: rows.append({'feature': mm.group(1), 'contribution': float(mm.group(3)) * (1 if mm.group(2) == '+' else -1)})
    return pd.DataFrame(rows)


def timeline(h):
    view = h.tail(600)
    lo, hi = int(view['window'].min()), int(view['window'].max())
    bands = (view[view['true'] != 'Benign'].groupby('segment')
             .agg(start=('window', 'min'), end=('window', 'max'), attack=('true', 'first')).reset_index())
    bands['end'] = bands['end'] + 1
    al = pd.DataFrame([a for a in ss.alerts if lo <= a['window'] <= hi and a['event'] != 'CLEARED'])
    x = alt.X('window:Q', title='Window', scale=alt.Scale(domain=[lo, hi + 1]))
    layers = []
    if len(bands):
        layers.append(alt.Chart(bands).mark_rect(color=BAND, opacity=0.18).encode(
            x='start:Q', x2='end:Q', tooltip=[alt.Tooltip('attack:N', title='True attack')]))
    layers.append(alt.Chart(view).mark_line(color=BLUE, strokeWidth=2).encode(
        x=x, y=alt.Y('score:Q', title='P(attack)', scale=alt.Scale(domain=[0, 1])),
        tooltip=['window', alt.Tooltip('score:Q', format='.2f', title='P(attack)'),
                 alt.Tooltip('pred:N', title='Predicted'), alt.Tooltip('true:N', title='True')]))
    if thr is not None:
        layers.append(alt.Chart(pd.DataFrame({'t': [thr]})).mark_rule(color='#52514e', strokeDash=[4, 3], strokeWidth=1)
                      .encode(y='t:Q', tooltip=[alt.Tooltip('t:Q', title='Alert threshold')]))
    if len(al):
        layers.append(alt.Chart(al).mark_rule(strokeWidth=2).encode(
            x='window:Q',
            color=alt.Color('severity:N', scale=alt.Scale(domain=['HIGH', 'MEDIUM'], range=[CRITICAL, SERIOUS]),
                            legend=alt.Legend(title='Alert', orient='top')),
            tooltip=['window', 'event', 'type', 'severity', 'confidence', 'true']))
    return alt.layer(*layers).properties(height=250)


# ---------------------------------------------------------------- tabs
tab_live, tab_perf, tab_xai, tab_zoo, tab_why, tab_res, tab_sys = st.tabs(
    ['🛰️ Live monitor', '📊 Performance', '🧠 Explain', '🏆 Model comparison', '🔬 Why leakage-free?',
     '📑 Research results', '⚙️ System & models'])

with tab_live:
    @st.fragment(run_every=1.0 if ss.get('playing') else None)
    def live():
        if ss.get('playing'): advance(speed)
        h = pd.DataFrame(ss.hist)
        active = ss.ids.alerts.active
        prog = f'{ss.pos:,} / {len(ss.recs):,} windows'
        if active:
            sev = ss.ids.alerts.severity(active)
            st.error(f'{SEV_ICON.get(sev, "🔴")} **ACTIVE ALERT — {active}** · severity {sev} · {prog}')
        elif len(h):
            st.success(f'✅ Normal traffic — no active alert · {prog}')
        else:
            st.info('Press ▶ Play (or ⏭ Step) to start replaying traffic.')
        if not len(h): return
        raised = [a for a in ss.alerts if a['event'] == 'RAISED']
        n_att, det, dly = segment_stats(h)
        k = st.columns(6)
        k[0].metric('Windows', f'{len(h):,}')
        real = sum(a['true'] != 'Benign' for a in raised)
        k[1].metric('Alerts raised', len(raised), delta=f'{real / len(raised) * 100:.0f}% real' if raised else None,
                    delta_color='off')
        k[2].metric('False alarms', sum(a['true'] == 'Benign' for a in raised))
        k[3].metric('Attacks detected', f'{det}/{n_att}' if n_att else '–')
        k[4].metric('Median delay (win)', f'{dly:.0f}' if dly is not None else '–')
        k[5].metric('Median latency', f"{h['latency'].median():.2f} ms")
        st.altair_chart(timeline(h), width='stretch')
        st.caption('Blue line = P(attack) · dashed line = alert threshold · grey band = true attack segment · '
                   'vertical rule = alert raised (red HIGH, orange MEDIUM). Last 600 windows.')
        left, right = st.columns([2, 3])
        with left:
            st.subheader('Latest alert')
            last = next((a for a in reversed(ss.alerts) if a['event'] != 'CLEARED'), None)
            if last:
                color = SEV_COLOR.get(last['severity'], CRITICAL)
                ok = '✅ correct type' if last['type'] == last['true'] else f"⚠️ true: {last['true']}"
                st.markdown(f"""<div class="alert-card" style="--sev:{color}">
                    <b>{SEV_ICON.get(last['severity'], '')} {last['type']}</b> · {last['severity']} · {last['event']}<br>
                    window {last['window']} · confidence {last['confidence']:.2f} · flagged {last['flagged']} · {ok}
                    </div>""", unsafe_allow_html=True)
                r = parse_reasons(last['reasons'])
                if len(r):
                    st.caption('Why? Top SHAP contributions (per-window model)')
                    st.altair_chart(alt.Chart(r).mark_bar(cornerRadiusEnd=4, height=16).encode(
                        x=alt.X('contribution:Q', title='SHAP contribution (log-odds)'),
                        y=alt.Y('feature:N', sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                        color=alt.condition('datum.contribution > 0', alt.value(BLUE), alt.value(CRITICAL)),
                        tooltip=['feature', alt.Tooltip('contribution:Q', format='+.2f')]
                    ).properties(height=110), width='stretch')
            else:
                st.write('No alert yet.')
        with right:
            st.subheader('Alert log')
            if ss.alerts:
                log = pd.DataFrame(ss.alerts[::-1])
                log['why'] = log['reasons'].map(lambda r: ', '.join(r))
                st.dataframe(log.drop(columns=['reasons', 'x', 'ctx'], errors='ignore'), width='stretch', height=260, hide_index=True)
            else:
                st.write('Empty.')
    live()

with tab_perf:
    @st.fragment(run_every=3.0 if ss.get('playing') else None)
    def perf():
        h = pd.DataFrame(ss.hist)
        if len(h) < 50:
            st.info('Replay at least 50 windows first.'); return
        from sklearn.metrics import accuracy_score, f1_score
        present = sorted(set(h['true']))
        c = st.columns(3)
        c[0].metric('Window accuracy', f"{accuracy_score(h['true'], h['pred']) * 100:.1f}%")
        c[1].metric('Macro-F1 (classes seen)', f"{f1_score(h['true'], h['pred'], labels=present, average='macro'):.3f}")
        c[2].metric('Windows evaluated', f'{len(h):,}')
        f1 = f1_score(h['true'], h['pred'], labels=present, average=None)
        pc = pd.DataFrame({'class': present, 'F1': f1}).sort_values('F1')
        l, r = st.columns(2)
        with l:
            st.subheader('Per-class F1')
            base = alt.Chart(pc).encode(y=alt.Y('class:N', sort=None, title=None, axis=alt.Axis(labelLimit=220)),
                                        x=alt.X('F1:Q', scale=alt.Scale(domain=[0, 1])))
            st.altair_chart((base.mark_bar(color=BLUE, cornerRadiusEnd=4, height=12)
                             + base.mark_text(align='left', dx=4, fontSize=11).encode(text=alt.Text('F1:Q', format='.2f')))
                            .properties(height=max(200, 22 * len(pc))), width='stretch')
        with r:
            st.subheader('Confusion by attack family (row %)')
            ft, fp = h['true'].map(family), h['pred'].map(family)
            cm = pd.crosstab(ft, fp, normalize='index').mul(100).round(1).stack().rename('pct').reset_index()
            cm.columns = ['True', 'Predicted', 'pct']
            hm = alt.Chart(cm).encode(x=alt.X('Predicted:N'), y=alt.Y('True:N'))
            st.altair_chart((hm.mark_rect(stroke='white', strokeWidth=2).encode(
                color=alt.Color('pct:Q', scale=alt.Scale(range=['#cde2fb', '#0d366b']), title='%'),
                tooltip=['True', 'Predicted', alt.Tooltip('pct:Q', format='.1f')])
                + hm.mark_text(fontSize=11).encode(text=alt.Text('pct:Q', format='.0f'),
                                                   color=alt.condition('datum.pct > 50', alt.value('white'), alt.value('black'))))
                            .properties(height=300), width='stretch')
        st.caption('Hardest cases: DoS vs DDoS of the same protocol (same family, different attacker count) '
                   'and low-volume Recon (VulScan) — consistent with the paper.')
    perf()

with tab_xai:
    st.subheader('Why did the IDS raise this alert?')
    raised = [a for a in ss.alerts if a['event'] != 'CLEARED' and 'x' in a]
    if not raised:
        st.info('Replay some traffic in the Live monitor first; every raised alert can be explained here.')
    else:
        opts = {f"window {a['window']} · {a['type']} · {a['event']} (true: {a['true']})": a for a in reversed(raised)}
        a = opts[st.selectbox('Alert', list(opts))]
        mdl = st.radio('Explain which model', ['Per-window model', 'Context model (last 10 windows)'], horizontal=True)
        dfx, base = ss.ids.explainer.explain_full(np.array(a['x'], dtype='float32'), a['type'],
                                                  ctx=np.array(a['ctx'], dtype='float32'),
                                                  model='context' if mdl.startswith('Context') else 'window')
        top = dfx.head(12).copy(); top['label'] = top.apply(lambda r: f"{r['feature']} = {r['value']:.4g}", axis=1)
        st.altair_chart(alt.Chart(top).mark_bar(cornerRadiusEnd=4, height=14).encode(
            x=alt.X('contribution:Q', title=f"SHAP contribution towards '{a['type']}' (log-odds)"),
            y=alt.Y('label:N', sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.condition('datum.contribution > 0', alt.value(BLUE), alt.value(CRITICAL)),
            tooltip=['feature', alt.Tooltip('value:Q', format='.4g'), alt.Tooltip('contribution:Q', format='+.3f')]
        ).properties(height=330), width='stretch')
        st.caption(f'Base value {base:.2f} log-odds. Blue pushes towards the alerted class, red against. '
                   'Exact TreeSHAP (XGBoost built-in), so contributions add up to the model output.')
    st.divider()
    st.subheader('Global explanation over the replayed traffic')
    n_seen = len(ss.hist)
    if n_seen < 200:
        st.info('Replay at least 200 windows to compute global importance.')
    elif st.button(f'Compute global SHAP on {min(n_seen, 3000):,} replayed windows'):
        idx = np.arange(min(n_seen, 3000))
        Xg = np.vstack([ss.ids.guard.transform(ss.recs[i]) for i in idx]); yg = np.array([ss.labels[i] for i in idx])
        ov, pcl = ss.ids.explainer.global_importance(Xg, yg, 'window')
        g = (ov.head(15) / ov.sum() * 100).rename('share').reset_index().rename(columns={'index': 'feature'})
        c1, c2 = st.columns([2, 3])
        c1.altair_chart(alt.Chart(g).mark_bar(color=BLUE, cornerRadiusEnd=4, height=12).encode(
            x=alt.X('share:Q', title='share of mean |SHAP| (%)'), y=alt.Y('feature:N', sort=None, title=None),
            tooltip=['feature', alt.Tooltip('share:Q', format='.2f')]).properties(height=360, title='Top features (all classes)'), width='stretch')
        if pcl is not None:
            hm = pcl[ov.head(10).index]; hm = hm.div(hm.sum(axis=1), axis=0).mul(100).reset_index().melt('label', var_name='feature', value_name='pct')
            c2.altair_chart(alt.Chart(hm).mark_rect().encode(
                x=alt.X('feature:N', title=None, sort=list(ov.head(10).index)), y=alt.Y('label:N', title=None, axis=alt.Axis(labelLimit=220)),
                color=alt.Color('pct:Q', scale=alt.Scale(range=['#cde2fb', '#0d366b']), title='%'),
                tooltip=['label', 'feature', alt.Tooltip('pct:Q', format='.1f')]).properties(height=360, title='What drives each class'), width='stretch')

with tab_zoo:
    zoo = ROOT / 'data' / 'zoo'
    if not (zoo / 'zoo_all_runs.csv').exists():
        st.info('Copy the 4 files from Drive results_v12 into data/zoo/ to see the 12-model comparison.')
    else:
        st.subheader('12 models, same leakage-free features')
        Z = pd.read_csv(zoo / 'zoo_all_runs.csv')
        proto = st.radio('Test protocol', ['Mixed hospital-like stream (mean of L=10, 100)', 'Official test set (one window at a time)'], horizontal=True)
        metric = st.selectbox('Metric', ['macro_f1', 'accuracy', 'weighted_f1', 'macro_precision', 'macro_recall', 'roc_auc_macro_ovr'])
        if proto.startswith('Mixed'):
            T = pd.read_csv(zoo / 'zoo_stream_mean.csv'); T['name'] = T['model'] + '  [' + T['view'] + ']'
        else:
            T = Z[Z['protocol'] == 'S static'].copy(); T['name'] = T['model']
        T['kind'] = np.where(T['model'].str.contains('ours'), 'Ours',
                             np.where(T['model'].str.contains('GRU|Transformer|MLP|CNN'), 'Deep learning', 'Classical ML'))
        T = T.sort_values(metric, ascending=False)
        st.altair_chart(alt.Chart(T).mark_bar(cornerRadiusEnd=4, height=13).encode(
            x=alt.X(f'{metric}:Q', title=metric.replace('_', ' ') + ' (%)', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('name:N', sort=None, title=None, axis=alt.Axis(labelLimit=380, labelOverlap=False)),
            color=alt.Color('kind:N', scale=alt.Scale(domain=['Ours', 'Classical ML', 'Deep learning'], range=[ORANGE, BLUE, '#1baf7a']),
                            legend=alt.Legend(orient='top', title=None)),
            tooltip=['name', alt.Tooltip(f'{metric}:Q', format='.2f'), 'accuracy', 'macro_f1']
        ).properties(height=max(260, 26 * len(T))), width='stretch')
        st.dataframe(T[['name', 'accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall', 'roc_auc_macro_ovr']],
                     hide_index=True, width='stretch')
        st.caption('Single training seed. Boosting + context is best; our ensemble and LightGBM/XGBoost with context are within 0.5 '
                   'points (a tie). Deep-learning models are 10-14 points lower on this tabular data. Colab training/inference '
                   'times are not comparable across models (GPU vs CPU) and are therefore not shown.')
        pcf_p, cm_p = zoo / 'zoo_per_class_f1.csv', zoo / 'zoo_confusion.npz'
        if pcf_p.exists() and cm_p.exists():
            PCF = pd.read_csv(pcf_p, index_col=0)
            default = next((k for k in PCF.index if 'ours' in k and 'L=100' in k), PCF.index[0])
            key = st.selectbox('Inspect one run', list(PCF.index), index=list(PCF.index).index(default))
            z = np.load(cm_p, allow_pickle=True); keys = list(z['keys']); labs = list(z['labels'])
            c1, c2 = st.columns(2)
            pf = PCF.loc[key].rename('F1').reset_index().rename(columns={'index': 'class'}).sort_values('F1')
            c1.altair_chart(alt.Chart(pf).mark_bar(color=BLUE, cornerRadiusEnd=4, height=11).encode(
                x=alt.X('F1:Q', scale=alt.Scale(domain=[0, 100])), y=alt.Y('class:N', sort=None, title=None, axis=alt.Axis(labelLimit=220)),
                tooltip=['class', alt.Tooltip('F1:Q', format='.1f')]).properties(height=420, title='Per-class F1 (%)'), width='stretch')
            if key in keys:
                cm = z['cms'][keys.index(key)].astype(float); cm = cm / cm.sum(1, keepdims=True).clip(min=1) * 100
                cmd = pd.DataFrame(cm, index=labs, columns=labs).stack().rename('pct').reset_index(); cmd.columns = ['True', 'Predicted', 'pct']
                c2.altair_chart(alt.Chart(cmd).mark_rect().encode(
                    x=alt.X('Predicted:N', axis=alt.Axis(labelAngle=-60, labelLimit=140)), y=alt.Y('True:N', axis=alt.Axis(labelLimit=180)),
                    color=alt.Color('pct:Q', scale=alt.Scale(range=['#ffffff', '#0d366b']), title='row %'),
                    tooltip=['True', 'Predicted', alt.Tooltip('pct:Q', format='.1f')]).properties(height=420, title='Confusion matrix (row %)'), width='stretch')

with tab_why:
    st.subheader('One feature, IAT, gives 98.9% accuracy — because it is the capture timestamp')
    st.markdown('In CICIoMT2024 the **IAT** column does not hold inter-arrival times. Multiplying it by the window '
                'size (k = 10 or 20) gives the **Unix time at which the file was recorded** (Aug–Sep 2023). Every attack '
                'class was captured in its own session, and the official *test* file is simply the end of the same '
                'session — so IAT tells the model *when* traffic was recorded, which reveals *what* was recorded.')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('IAT alone', '98.9%', help='XGBoost, 19 classes, train -> official test')
    c2.metric('Best other single feature', '63.8%', help='Min; median of all 45 features: 13.7%')
    c3.metric('Published CNN without artifacts', '−20.9 pts', help='98.48% -> 77.63% accuracy (faithful re-implementation)')
    c4.metric('Files that decode to a real date', '72 / 72', help='51 train + 21 test CSV files')
    dec_path = ROOT / 'data' / 'iat_decode_ciciomt2024.csv'
    if dec_path.exists():
        dec = pd.read_csv(dec_path, parse_dates=['decoded_start_utc', 'decoded_end_utc'])
        dec['family'] = dec['label'].map(lambda l: 'Spoofing' if 'Spoof' in l else ('Benign' if l == 'Benign' else l.split('-')[0]))
        dec['end'] = dec[['decoded_end_utc', 'decoded_start_utc']].apply(
            lambda r: max(r.iloc[0], r.iloc[1] + pd.Timedelta(minutes=40)), axis=1)
        order = dec.groupby('label')['decoded_start_utc'].min().sort_values().index.tolist()
        st.altair_chart(alt.Chart(dec).mark_bar(height=9, cornerRadius=2).encode(
            x=alt.X('decoded_start_utc:T', title='Decoded capture time (UTC) = IAT × k'), x2='end:T',
            y=alt.Y('label:N', sort=order, title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False, labelFontSize=11)),
            color=alt.Color('family:N', scale=alt.Scale(domain=['DDoS', 'DoS', 'MQTT', 'Recon', 'Spoofing', 'Benign'],
                            range=[BLUE, ORANGE, '#1baf7a', '#eda100', '#e87ba4', BAND]), legend=alt.Legend(orient='top', title=None)),
            opacity=alt.Opacity('split:N', scale=alt.Scale(domain=['train', 'test'], range=[1.0, 0.5]), legend=alt.Legend(orient='top', title=None)),
            tooltip=['file', 'split', alt.Tooltip('decoded_start_utc:T', format='%d %b %H:%M', title='start'),
                     alt.Tooltip('decoded_end_utc:T', format='%d %b %H:%M', title='end'), 'k_window']
        ).properties(height=520), width='stretch')
        st.caption('Each bar = one CSV file of CICIoMT2024 (faded = official test file). One capture session per class.')
    with st.expander('How the bug happens (CIC pcap2csv feature extractor)'):
        st.markdown("""
1. Packets are read in order; an older code path computes `IAT = ts - first_pac_time`.
2. Every 20 packets `first_pac_time` is reset to **0**, so the next packet gets `IAT = ts` — the full Unix time (~1.69e9).
3. Rows are then averaged over a window of 10 or 20 packets → **IAT ≈ Unix time ÷ k** (e.g. 8.47e7 × 20 = 6 Sep 2023).
4. **Control:** in the sister dataset CICIoT2023 (same lab), IAT holds genuine inter-arrival times (median 7 ms benign,
   30 µs floods) — so the CICIoMT2024 value is an extraction error, not a design choice.

Check any CIC-style dataset yourself: `python scripts/audit_dataset.py <folder with CSVs>`""")
    long = AUDIT.melt('setting', var_name='metric', value_name='value')
    bars = alt.Chart(long).encode(
        y=alt.Y('setting:N', sort=list(AUDIT['setting']), title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X('value:Q', title='%', scale=alt.Scale(domain=[0, 100])),
        yOffset='metric:N',
        color=alt.Color('metric:N', scale=alt.Scale(domain=['Accuracy', 'Macro-F1'], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient='top', title=None)),
        tooltip=['setting', 'metric', alt.Tooltip('value:Q', format='.2f')])
    st.altair_chart((bars.mark_bar(cornerRadiusEnd=4, height=11)
                     + bars.mark_text(align='left', dx=4, fontSize=11).encode(text=alt.Text('value:Q', format='.1f'),
                                                                              color=alt.value('gray')))
                    .properties(height=360), width='stretch')
    st.caption('Official CICIoMT2024 test set; mean over seeds where stated.')

with tab_res:
    st.subheader('Leakage-free model: what is left, and why')
    a, b = st.columns(2)
    with a:
        st.markdown('**Label granularity** (static official test)')
        st.dataframe(pd.DataFrame([('Binary (attack vs benign)', 99.64, 95.79), ('6 families (official)', 72.31, 75.90),
                                   ('13 types (DoS+DDoS merged)', 99.33, 82.07), ('19 classes', 71.66, 69.16)],
                                  columns=['Level', 'Accuracy %', 'Macro-F1 %']), hide_index=True, width='stretch')
        st.caption('97.6% of all errors are DoS <-> DDoS swaps of the same protocol (attacker count is not visible in a window).')
    with b:
        st.markdown('**Generalisation**')
        st.dataframe(pd.DataFrame([('Unseen recording, 13 types', 99.86), ('Unseen recording, 19 classes', 74.26),
                                   ('IAT only, unseen recording', 89.94), ('Zero-day Recon (leakage-free / all 42)', '96.1 / 76.5'),
                                   ('Zero-day Spoofing (leakage-free / all 42)', '71.6 / 84.6')],
                                  columns=['Test', 'Result %']).astype(str), hide_index=True, width='stretch')
    st.subheader('Alerting at a realistic base rate (95% benign traffic, Lite model, 3-of-5)')
    st.dataframe(pd.DataFrame([('Argmax (no threshold)', 5.79, 17.1, 52.4, 99.9),
                               ('Threshold 0.95 (chosen on validation)', 0.37, 72.2, 93.9, 96.5)],
                              columns=['Rule', 'False alarms / 1000 benign', 'Alert precision %', 'Window precision %',
                                       'Attack segments detected %']), hide_index=True, width='stretch')
    st.subheader('Temporal context vs simple smoothing (mixed streams, macro-F1 %, mean over L)')
    st.dataframe(pd.DataFrame([('Per-window', 67.66), ('Per-window + rolling average (K=5)', 61.91),
                               ('Per-window + rolling average (K=10)', 53.28), ('Context ensemble (ours)', 72.57)],
                              columns=['Method', 'Macro-F1 %']), hide_index=True, width='stretch')
    st.subheader('Is window size a second leak? (v14)')
    st.dataframe(pd.DataFrame([('Window-size-only oracle (knows only flood vs low-rate group)', '24.76 / 5.72'),
                               ('Errors of our model that cross the two groups', '0.26% of errors'),
                               ('Low-rate group: accuracy / when the group is given', '86.72 / 86.73'),
                               ('Flood group: accuracy / when the group is given', '70.99 / 71.00')],
                              columns=['Check', 'Accuracy % / Macro-F1 %']), hide_index=True, width='stretch')
    st.caption('Window size can only separate floods from low-rate classes. The model already never mixes the two groups, '
               'so its accuracy comes from telling apart classes with the same window size.')
    st.caption('All numbers from the accompanying research notebooks (v10a–v10f, v14); see RESEARCH_LOG_CICIoMT2024.md.')

with tab_sys:
    st.subheader('Pipeline')
    st.graphviz_chart("""digraph {
        rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6", fontname="Helvetica"];
        edge [color="#52514e"];
        IN [label="Traffic window\\n(45 CIC features)", fillcolor="#f0efec", color="#9a9890"];
        G [label="M1 Artifact guard\\n−IAT (timestamp)\\n−Number, −Weight, −3 redundant"];
        C [label="M3 Context buffer\\nlast K=10 windows\\nmean + std"];
        A [label="Per-window XGBoost"]; B [label="Context XGBoost"];
        F [label="Fusion\\nw·P_a + (1−w)·P_c"];
        AL [label="M6 Alert manager\\nP(attack) ≥ t, m-of-n, severity"]; X [label="M7 SHAP reasons"];
        OUT [label="Alert + explanation", fillcolor="#fde8e8", color="#d03b3b"];
        IN -> G -> C; G -> A; C -> B; A -> F; B -> F; F -> AL -> OUT; X -> OUT; A -> X [style=dashed];
    }""", width='stretch')
    st.subheader('Model variants (v9b, Colab CPU, 1 thread)')
    st.dataframe(VARIANTS, hide_index=True, width='stretch')
    info = cfg.get('model_info', {})
    st.markdown(f"**Active model:** `{model}` · {len(cfg['features'])} features · K = {cfg['context_k']} · "
                f"fusion w = {cfg['fusion_w']} · smoothing {m}-of-{n} · threshold {thr if thr is not None else 'argmax'}"
                + (f" · depth {info.get('max_depth')}, ≤{info.get('max_trees')} trees" if info else ''))
    with st.expander('Features used by the model'):
        st.write(', '.join(cfg['features']))
