"""Streaming pipeline == batch predictions (consistency check).  Run:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from iomt_ids import (AlertManager, ArtifactGuard, IDSPipeline, alert_metrics, available_models,   # noqa: E402
                      load_config)
from iomt_ids.audit import decode_iat_values   # noqa: E402


def test_streaming_matches_batch():
    cfg = load_config(ROOT / 'config.yaml', available_models(ROOT / 'config.yaml')[0])
    df = pd.read_parquet(cfg['demo_stream']).head(1500)
    ids = IDSPipeline(cfg)
    X = ids.guard.transform_frame(df)
    batch = np.array(cfg['labels'])[ids.detector.predict_proba_batch(X, cfg['context_k']).argmax(1)]
    recs = df.drop(columns=[c for c in ('label', 'segment') if c in df]).to_dict('records')
    stream = np.array([ids.process(i, r).predicted for i, r in enumerate(recs)])
    assert (stream == batch).mean() > 0.99


def test_guard_blocks_artifacts():
    g = ArtifactGuard(['Rate', 'TCP'], ['IAT', 'Number', 'Weight'], ['Drate'])
    x = g.transform({'Rate': 1.0, 'TCP': 0.0, 'IAT': 84_696_985.0, 'Number': 9.5})
    assert x.tolist() == [1.0, 0.0]


def test_alert_smoothing():
    am = AlertManager(m=3, n=5)
    events = [am.update(i, l, 0.9) for i, l in enumerate(['Benign', 'DDoS-UDP', 'DDoS-UDP', 'DDoS-UDP', 'Benign'])]
    assert [e.event for e in events if e] == ['RAISED']


def test_threshold_reduces_flags():
    cfg = load_config(ROOT / 'config.yaml', available_models(ROOT / 'config.yaml')[0])
    df = pd.read_parquet(cfg['demo_stream']).head(800)
    recs = df.drop(columns=[c for c in ('label', 'segment') if c in df]).to_dict('records')
    ids = IDSPipeline(cfg); ids.set_threshold(None)
    loose = sum(ids.process(i, r).flagged for i, r in enumerate(recs))
    ids = IDSPipeline(cfg); ids.set_threshold(0.99)
    strict = sum(ids.process(i, r).flagged for i, r in enumerate(recs))
    assert strict <= loose


def test_alert_metrics_simple():
    y = ['Benign'] * 10 + ['DDoS-UDP'] * 10 + ['Benign'] * 10
    seg = [0] * 10 + [1] * 10 + [2] * 10
    flags = [False] * 10 + [True] * 10 + [False] * 10
    m = alert_metrics(y, seg, flags, 3, 5)
    assert m['attack_segments_detected_%'] == 100.0 and m['median_delay_windows'] == 2.0 and m['event_precision_%'] == 100.0


def test_iat_timestamp_decode():
    ts = 1_693_900_000 + np.arange(100) * 2.0            # Sep 2023
    out = decode_iat_values(np.r_[ts / 20, np.full(100, 3e-5)])
    assert out['timestamp_like'] and out['k_window'] == 20 and out['decoded_start_utc'].startswith('2023-09')
    assert not decode_iat_values(np.full(100, 3e-5))['timestamp_like']


def test_explainer_full_and_global():
    cfg = load_config(ROOT / 'config.yaml', available_models(ROOT / 'config.yaml')[0])
    df = pd.read_parquet(cfg['demo_stream']).head(300)
    ids = IDSPipeline(cfg); X = ids.guard.transform_frame(df)
    dfx, base = ids.explainer.explain_full(X[0], cfg['labels'][0])
    assert len(dfx) == len(cfg['features']) and np.isfinite(base)
    ov, pc = ids.explainer.global_importance(X, df['label'].to_numpy())
    assert len(ov) == len(cfg['features']) and pc is not None
