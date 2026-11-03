"""IoMT-IDS Zero-Day Lab — detect, group, label and learn an attack the model has never seen.

Run:  streamlit run app/zero_day_lab.py        (the main dashboard stays: streamlit run app/dashboard.py)
Needs data/zeroday/reference_train.parquet + reference_val.parquet (exported by notebooks/v13_zero_day_continual.py).
"""
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from iomt_ids.continual import ModelRegistry                       # noqa: E402
from iomt_ids.drift import check_raw_capture, drift_report, summarize  # noqa: E402
from iomt_ids.patterns import project_2d                           # noqa: E402
from iomt_ids.zero_day import ZeroDayLab, scenarios                # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#9a9890'
CATS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, '#8e6fd8']
STATUS = {'Unknown pattern': ORANGE, 'Known-attack alert': BLUE, 'Passed': '#c9c7c0'}
ZD = ROOT / 'data' / 'zeroday'
LAB_DIR = ROOT / 'models' / 'lab'

st.set_page_config(page_title='IoMT-IDS Zero-Day Lab', page_icon='🧪', layout='wide')
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: 10px 14px;}
.chip {display:inline-block; padding:2px 10px; margin:0 6px 6px 0; border-radius:999px;
       border:1px solid rgba(128,128,128,.35); font-size:.82rem;}
.step {font-size:.8rem; font-weight:600; letter-spacing:.04em; color:#2a78d6; text-transform:uppercase; margin-bottom:-.4rem;}
</style>""", unsafe_allow_html=True)
st.title('🧪 IoMT-IDS — Zero-Day Lab')
st.markdown('<span class="chip">Hide an attack from training</span><span class="chip">Open-set detection: kNN + energy</span>'
            '<span class="chip">Pattern discovery (KMeans)</span><span class="chip">Analyst labelling</span>'
            '<span class="chip">Continual learning with replay</span><span class="chip">Drift monitor (PSI)</span>',
            unsafe_allow_html=True)
st.caption('Simulation on real CICIoMT2024 traffic: the lab model is trained on the reference set WITHOUT the chosen attack, '
           'then the demo stream (official test set) is replayed. Window-level model only, so absolute scores are lower '
           'than the main context-aware dashboard.')


@st.cache_data(show_spinner=False)
def load_data():
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    ref, val = pd.read_parquet(ZD / 'reference_train.parquet'), pd.read_parquet(ZD / 'reference_val.parquet')
    stream = pd.read_parquet(ROOT / cfg['demo_stream'])
    feats = [c for c in ref.columns if c != 'label' and c in stream.columns]
    return ref, val, stream, feats, sorted(ref['label'].unique())


have_ref = (ZD / 'reference_train.parquet').exists() and (ZD / 'reference_val.parquet').exists()
ss = st.session_state

with st.sidebar:
    st.header('Lab settings')
    if have_ref:
        ref, val, stream, FEATS, LABELS = load_data(); SC = scenarios(LABELS)
        scen = st.selectbox('Attack the model has NEVER seen', list(SC), help='These classes are removed from training.')
        st.caption('Hidden: ' + ', '.join(SC[scen]))
    method = st.radio('Unknown-detection score', ['knn+energy', 'knn', 'energy'], horizontal=True,
                      help='kNN = distance to known traffic · energy = no known class is confident · combined = either')
    fpr = st.slider('Allowed false-unknown rate on known traffic', 0.01, 0.10, 0.05, 0.01,
                    help='Threshold = this quantile on KNOWN validation windows.')
    thr = st.slider('P(attack) alert threshold', 0.50, 0.99, 0.95, 0.01)
    sustain = st.checkbox('Unknown only if sustained (3 of last 5 windows)', value=True,
                          help='Isolated novelty hits are mostly benign noise; real attacks last several windows.')
    incl = st.checkbox('Also group windows alerted as a known attack', value=False,
                       help='An unseen attack is often confidently mistaken for a known one; include those in discovery.')
    mode = st.radio('Learning mode', ['replay', 'no-replay', 'retrain'],
                    help='replay = fine-tune on new + replay buffer (default) · no-replay = forgets old classes · retrain = from scratch')
    if st.button('↺ Reset lab', width='stretch'):
        for k in ('lab', 'lab_key', 'base', 'bst', 'S0', 'S', 'pat', 'hist'): ss.pop(k, None)
        st.rerun()

tab_lab, tab_res, tab_drift = st.tabs(['🧪 Zero-day lab', '📑 Research results (v13)', '📉 Drift & data check'])


def status_chart(S):
    g = S.assign(group=np.where(S['hidden'], S['true'] + '  (unseen)', np.where(S['true'] == 'Benign', 'Benign', 'Other known attacks')))
    d = g.groupby(['group', 'status']).size().rename('n').reset_index()
    d['pct'] = d['n'] / d.groupby('group')['n'].transform('sum') * 100
    order = sorted(d['group'].unique(), key=lambda s: (not s.endswith('(unseen)'), s))
    return alt.Chart(d).mark_bar(stroke='white', strokeWidth=1).encode(
        x=alt.X('pct:Q', stack='normalize', title='share of windows', axis=alt.Axis(format='%')),
        y=alt.Y('group:N', sort=order, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False), scale=alt.Scale(paddingInner=0.35)),
        color=alt.Color('status:N', scale=alt.Scale(domain=list(STATUS), range=list(STATUS.values())), legend=alt.Legend(orient='top', title=None)),
        order=alt.Order('status:N', sort='descending'),
        tooltip=['group', 'status', 'n', alt.Tooltip('pct:Q', format='.1f')]).properties(height=alt.Step(30))


def metric_row(m, prev=None):
    c = st.columns(5)
    items = [('Unseen windows caught', 'unseen_caught_%', 1), ('…flagged UNKNOWN', 'unseen_flagged_unknown_%', 1),
             ('…alerted as a known attack', 'unseen_alerted_as_known_%', 1), ('Unseen missed', 'unseen_missed_%', -1),
             ('Benign false-unknown', 'benign_false_unknown_%', -1)]
    for col, (name, k, sign) in zip(c, items):
        v = m.get(k); dv = None if prev is None or prev.get(k) is None or v is None else v - prev[k]
        col.metric(name, '—' if v is None else f'{v:.1f}%', None if dv is None else f'{dv:+.1f} pts',
                   delta_color='normal' if sign > 0 else 'inverse')


# ====================================================================== LAB
with tab_lab:
    if not have_ref:
        st.warning('Reference data missing. Run notebooks/v13_zero_day_continual.py in Colab and copy '
                   '`reference_train.parquet` and `reference_val.parquet` into `data/zeroday/`.')
        st.stop()
    key = (scen, method, fpr, thr, sustain)
    if ss.get('lab_key') != key:
        for k in ('base', 'bst', 'S0', 'S', 'pat', 'hist'): ss.pop(k, None)
        ss.lab = ZeroDayLab(ref, val, stream, FEATS, LABELS, SC[scen], cache_dir=LAB_DIR / 'cache',
                            method=method, target_fpr=fpr, attack_threshold=thr,
                            unknown_smoothing=(3, 5) if sustain else None)
        ss.lab_key, ss.hist = key, []
    lab = ss.lab

    st.markdown('<div class="step">Step 1</div>', unsafe_allow_html=True)
    st.subheader(f'Train a model that has never seen: {scen}')
    if 'base' not in ss:
        if st.button('Train lab model', type='primary'):
            with st.spinner(f'Training XGBoost on {len(lab.Xr):,} reference windows without {scen}…'):
                t = time.time(); b, cached = lab.train_base(); lab.fit_novelty(b)
                ss.base = ss.bst = b; ss.S0 = ss.S = lab.score(b)
                ss.ev0 = lab.evaluate(b); ss.train_info = 'loaded from cache' if cached else f'trained in {time.time() - t:.1f} s'
            st.rerun()
        st.stop()
    st.success(f'Lab model ready ({ss.train_info}). Known classes: {len(lab.known)} · hidden: {len(lab.hidden)} · '
               f'novelty threshold = {fpr:.0%} false-unknown on known validation windows.')

    st.markdown('<div class="step">Step 2</div>', unsafe_allow_html=True)
    st.subheader('Replay traffic: what happens to the unseen attack?')
    metric_row(lab.summary(ss.S0))
    st.altair_chart(status_chart(ss.S0), width='stretch')
    st.caption(f'All {len(ss.S0):,} replayed windows (before learning). Orange = flagged as an unknown pattern · blue = alerted '
               'as some known attack (detected, but wrong type) · grey = passed as normal.')

    st.markdown('<div class="step">Step 3</div>', unsafe_allow_html=True)
    st.subheader('Discovered patterns (discovery half of the stream)')
    if 'pat' not in ss:
        ss.pat = lab.discover(ss.S, include_alerts=incl)
    idx, plab, summ, sil = ss.pat
    if not len(summ):
        st.info('Too few unknown windows to form a pattern. Try “Also group windows alerted as a known attack” '
                'or a higher false-unknown rate.')
    else:
        Z2 = project_2d(np.clip(lab.novelty.space(lab.Xs[idx]), -6, 6))   # clip: one extreme feature would flatten the view
        pts = pd.DataFrame({'x': Z2[:, 0], 'y': Z2[:, 1], 'pattern': np.where(plab >= 0, ['P' + str(p + 1) for p in plab], 'noise'),
                            'status': ss.S['status'].to_numpy()[idx]})
        pts = pts.sample(min(len(pts), 4000), random_state=0)
        doms = [f'P{i + 1}' for i in range(len(summ))] + ['noise']
        show = summ[['pattern', 'windows', 'nearest_known', 'second_nearest', 'signature']].rename(
            columns={'nearest_known': 'nearest known class', 'second_nearest': '2nd nearest', 'signature': 'signature vs benign traffic'})
        st.dataframe(show, hide_index=True, width='stretch',
                     column_config={'signature vs benign traffic': st.column_config.TextColumn(width='large')})
        c1, c2 = st.columns([3, 2])
        c1.altair_chart(alt.Chart(pts).mark_circle(size=22, opacity=0.65, stroke='white', strokeWidth=0.4).encode(
            x=alt.X('x:Q', title='PC 1', axis=alt.Axis(labels=False, ticks=False)), y=alt.Y('y:Q', title='PC 2', axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color('pattern:N', scale=alt.Scale(domain=doms, range=CATS[:len(doms) - 1] + [GREY]), legend=alt.Legend(orient='top', title=None)),
            tooltip=['pattern', 'status']).properties(height=300, title='Flagged windows (PCA of novelty space)'), width='stretch')
        c2.markdown('**How to read this**')
        c2.caption('Each dot is a flagged window from the discovery half. Colours are the patterns found by KMeans; '
                   '“noise” = groups too small to report. Signature = features furthest from normal (Benign) traffic, '
                   'in σ of the signed-log scale. Nearest known class = closest class centroid. '
                   'Unknown windows are deliberately over-clustered (k ≈ n/30): small pure patterns are easier to name, and '
                   'several patterns can get the same name.')
        with st.expander('Ground truth of each pattern (evaluation only — an analyst would not see this)'):
            st.dataframe(summ[['pattern', 'windows', 'true_composition']], hide_index=True, width='stretch')

        st.markdown('<div class="step">Step 4</div>', unsafe_allow_html=True)
        st.subheader('Analyst: name a pattern and teach the IDS')
        a, b, c = st.columns([2, 2, 1])
        pids = a.multiselect('Pattern(s) — select several if they are the same attack', list(summ['pattern']),
                             default=list(summ['pattern'])[:1])
        sel_ids = summ[summ['pattern'].isin(pids)]['id'].tolist(); win = idx[np.isin(plab, sel_ids)]
        helper = c.toggle('Demo helper', value=True, help='Pre-selects the true majority unseen class of the selected windows '
                          '(ground truth, for the demo only — a real analyst decides from the signature and raw traffic).')
        opts = lab.hidden + [l for l in lab.labels if l not in lab.hidden] + ['➕ New name…']
        tr_all = pd.Series(lab.true[win]); tr = tr_all[tr_all.isin(lab.hidden)]
        maj = tr_all.value_counts().index[0] if len(tr_all) else None
        default = opts.index(tr.value_counts().index[0]) if helper and len(tr) else 0
        if helper and maj is not None and maj not in lab.hidden:
            default = opts.index(maj) if maj in opts else 0
            st.warning(f'Demo helper: this pattern is mostly **{maj}** ({tr_all.eq(maj).mean():.0%}), i.e. already-known traffic — '
                       'a real analyst would dismiss it as a false alarm instead of teaching it as a new attack. '
                       'Teaching it under an attack name would poison the model (more false alarms on normal traffic).')
        name = b.selectbox('Label it as', opts, index=default)
        if name == '➕ New name…': name = b.text_input('New attack name', 'New-Attack-1').strip()
        c.markdown(f"{len(win):,} windows · mode **{mode}**")
        pid = '+'.join(pids)
        if st.button(f'Teach the IDS: {pid or "—"} → {name}', type='primary', disabled=not (name and len(win))):
            with st.spinner('Continual learning…'):
                ev_prev, s_prev = lab.evaluate(ss.bst), lab.summary(ss.S, 'evaluation')
                bst, info = lab.learn(ss.bst, win[:1000], name, mode)
                S_new = lab.score(bst); ev = lab.evaluate(bst); s_new = lab.summary(S_new, 'evaluation')
                reg = ModelRegistry(LAB_DIR)
                v = reg.save(bst, lab.labels, {**info, 'scenario': scen, 'pattern': pid},
                             {k: ev[k] for k in ('old_macro_f1', 'new_macro_f1', 'accuracy_%', 'unseen_recognized_%')} |
                             {'unseen_caught_%': s_new['unseen_caught_%']})
                ss.hist.append({'version': v, 'pattern': pid, 'label': name, 'mode': info['mode'], 'windows': info['n_windows'],
                                'train s': info['train_sec'], 'old-class F1 before': ev_prev['old_macro_f1'], 'old-class F1 after': ev['old_macro_f1'],
                                'unseen F1 before': ev_prev['new_macro_f1'], 'unseen F1 after': ev['new_macro_f1'],
                                'unseen recognized %': ev['unseen_recognized_%'],
                                '_s_prev': s_prev, '_s_new': s_new, '_ev_prev': ev_prev, '_ev': ev})
                ss.bst, ss.S = bst, S_new; ss.pop('pat', None)
            st.rerun()

    if ss.hist:
        st.markdown('<div class="step">Result</div>', unsafe_allow_html=True)
        st.subheader('Before vs after learning — evaluation half (segments never labelled)')
        h = ss.hist[-1]
        ev0 = ss.ev0
        c = st.columns(4)
        c[0].metric('Unseen-class macro-F1', f"{h['_ev']['new_macro_f1']:.1f}%", f"{h['_ev']['new_macro_f1'] - ev0['new_macro_f1']:+.1f} pts vs start")
        c[1].metric('Old-class macro-F1', f"{h['_ev']['old_macro_f1']:.1f}%", f"{h['_ev']['old_macro_f1'] - ev0['old_macro_f1']:+.1f} pts (forgetting if −)")
        c[2].metric('Unseen windows recognized', f"{h['_ev'].get('unseen_recognized_%') or 0:.1f}%",
                    help='Unseen-attack windows of the evaluation half now classified as a LEARNED class (any of them).')
        c[3].metric('Update time', f"{h['train s']:.1f} s")
        metric_row(h['_s_new'], h['_s_prev'])
        pc = pd.DataFrame({'before (start)': pd.Series(ev0['per_class_f1']), 'after': pd.Series(h['_ev']['per_class_f1'])})
        pc = pc.loc[[l for l in pc.index if l in lab.hidden or l in [lab.labels[i] for i in lab.learned]]]
        if len(pc):
            d = pc.fillna(0).reset_index(names='class').melt('class', var_name='when', value_name='F1')
            st.altair_chart(alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X('F1:Q', scale=alt.Scale(domain=[0, 100]), title='F1 (%)'), y=alt.Y('class:N', title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False), scale=alt.Scale(paddingInner=0.3)),
                yOffset=alt.YOffset('when:N', sort=['before (start)', 'after']), color=alt.Color('when:N', scale=alt.Scale(domain=['before (start)', 'after'], range=[GREY, ORANGE]),
                                                  legend=alt.Legend(orient='top', title=None)),
                tooltip=['class', 'when', alt.Tooltip('F1:Q', format='.1f')]).properties(height=alt.Step(15), title='Unseen / learned classes: F1 before vs after'),
                width='stretch')
        st.altair_chart(status_chart(ss.S[~ss.S['discovery']]), width='stretch')
        st.caption('Status of the evaluation-half windows with the updated model.')
        st.markdown('**Learning history (this session)** — every version is saved in `models/lab/` with its lineage.')
        st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in ss.hist]).round(2),
                     hide_index=True, width='stretch')
        st.caption('Label the next pattern to continue (several patterns may get the same name), or switch the learning mode '
                   'to “no-replay” to see catastrophic forgetting. DoS and DDoS of the same protocol usually fall into ONE pattern, '
                   'so naming it once recognises both as that name (they are not separable from window features).')

# ====================================================================== RESEARCH
with tab_res:
    files = {k: ZD / f for k, f in [('Z1', 'Z1_unknown_scores.csv'), ('Z2', 'Z2_system_view.csv'),
                                    ('Z3', 'Z3_pattern_discovery.csv'), ('Z4', 'Z4_continual_learning.csv')]}
    if not all(p.exists() for p in files.values()):
        st.info('Copy the v13 outputs (Z1–Z4 CSVs and fig_*.png) from Drive results_v13 into data/zeroday/.')
    else:
        Z1, Z2, Z3, Z4 = (pd.read_csv(p) for p in files.values())
        st.subheader('Z1 · How well does each score separate unseen attacks from known traffic? (official test set)')
        pv = Z1.pivot(index='scenario', columns='score', values='AUROC_%').round(1)
        st.dataframe(pv, width='stretch')
        d = Z1.groupby('score')[['AUROC_%', 'unknown_recall_%', 'known_test_false_unknown_%']].mean().round(1).reset_index()
        st.altair_chart(alt.Chart(d).mark_bar(color=BLUE, cornerRadiusEnd=4, height=14).encode(
            x=alt.X('AUROC_%:Q', scale=alt.Scale(domain=[0, 100]), title='mean AUROC over 8 scenarios (%)'),
            y=alt.Y('score:N', sort='-x', title=None, axis=alt.Axis(labelLimit=240)),
            tooltip=['score', 'AUROC_%', 'unknown_recall_%', 'known_test_false_unknown_%']).properties(height=200), width='stretch')
        st.subheader('Z2 · System view: is the unseen attack caught at all?')
        st.dataframe(Z2.round(1), hide_index=True, width='stretch')
        st.subheader('Z3 · Pattern discovery quality')
        st.dataframe(Z3.round(3), hide_index=True, width='stretch')
        st.subheader('Z4 · Continual learning (mean over scenarios)')
        agg = Z4.groupby(['strategy', 'n_per_class'])[['old_macro_f1', 'new_macro_f1', 'forgetting_pts', 'train_sec']].mean().round(2).reset_index()
        st.dataframe(agg, hide_index=True, width='stretch')
        for f in ('fig_zero_day_auroc.png', 'fig_continual_learning.png'):
            if (ZD / f).exists(): st.image(str(ZD / f))

# ====================================================================== DRIFT
with tab_drift:
    if not have_ref:
        st.info('Needs data/zeroday/reference_train.parquet.')
    else:
        st.subheader('Has the traffic moved away from the training distribution?')
        src = st.radio('Current traffic', ['Benign traffic only (like with like)', 'Per class (each class vs its training sample)',
                                           'Whole demo stream (all classes)', 'Upload a CSV (CIC format)'], horizontal=True)
        cur, ref_d = None, ref
        if src.startswith('Benign'):
            cur, ref_d = stream[stream['label'] == 'Benign'], ref[ref['label'] == 'Benign']
        elif src.startswith('Per class'):
            rows = []
            for l in sorted(set(stream['label']) & set(ref['label'])):
                sm = summarize(drift_report(ref[ref['label'] == l], stream[stream['label'] == l], FEATS))
                rows.append({'class': l, 'status': sm['overall'], 'features with major shift': sm['features_major'],
                             'moderate': sm['features_moderate'], 'worst feature': sm['worst_feature'], 'worst PSI': round(sm['worst_psi'], 2)})
            pc = pd.DataFrame(rows).sort_values('worst PSI', ascending=False)
            st.dataframe(pc, hide_index=True, width='stretch')
            st.caption('Official test windows vs training windows of the SAME class. Classes with a major shift are the ones a '
                       'model trained on the training capture will find hardest (compare ARP spoofing with its low F1).')
        elif src.startswith('Whole'):
            cur = stream
        else:
            up = st.file_uploader('CIC-style CSV (raw 45 columns are fine)', type='csv')
            if up is not None:
                raw = pd.read_csv(up); raw.columns = [str(c).strip().replace(' ', '_') for c in raw.columns]
                ts = check_raw_capture(raw)
                if ts is not None:
                    if ts['timestamp_like']:
                        st.error(f"IAT in this file is a capture TIMESTAMP (×{ts['k_window']} → {ts['decoded_start_utc']} … "
                                 f"{ts['decoded_end_utc']} UTC), not an inter-arrival time. The artifact guard drops it.")
                    else:
                        st.success(f"IAT looks like a genuine inter-arrival time (median {ts['median_IAT']:.3g}).")
                cur = raw
        if cur is not None:
            rep = drift_report(ref_d, cur, FEATS); sm = summarize(rep)
            c = st.columns(4)
            c[0].metric('Overall', sm['overall'].upper()); c[1].metric('Features: major shift', sm['features_major'])
            c[2].metric('Moderate shift', sm['features_moderate']); c[3].metric('Worst', f"{sm['worst_feature']}", f"PSI {sm['worst_psi']:.2f}", delta_color='off')
            top = rep.head(15)
            st.altair_chart(alt.Chart(top).mark_bar(cornerRadiusEnd=4, height=12).encode(
                x=alt.X('PSI:Q', title='Population Stability Index'), y=alt.Y('feature:N', sort=None, title=None, axis=alt.Axis(labelOverlap=False)),
                color=alt.Color('status:N', scale=alt.Scale(domain=['stable', 'moderate shift', 'major shift'], range=[AQUA, YELLOW, '#d03b3b']),
                                legend=alt.Legend(orient='top', title=None)),
                tooltip=['feature', alt.Tooltip('PSI:Q', format='.3f'), 'status']).properties(height=24 * len(top) + 40), width='stretch')
            st.caption('PSI < 0.10 stable · 0.10–0.25 moderate · > 0.25 major. Reference = training sample. Benign test traffic vs '
                       'benign training traffic is the fair check (stable); the whole stream shows a “major shift” only because its '
                       'class mix differs from the balanced reference — a reminder that PSI alarms need like-with-like comparison.')
