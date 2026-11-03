"""Demo stream pe batch evaluation + latency benchmark (laptop CPU).

Usage:  python scripts/evaluate.py                # compare all model variants that are present
        python scripts/evaluate.py --model lite
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from iomt_ids import IDSPipeline, alert_metrics, available_models, load_config   # noqa: E402


def evaluate(model):
    cfg = load_config(ROOT / 'config.yaml', model)
    print(f'\n===== Model: {model} =====')
    df = pd.read_parquet(cfg['demo_stream'])
    ids = IDSPipeline(cfg)
    X = ids.guard.transform_frame(df)
    y = df['label'].to_numpy()

    # 1) batch quality
    p = ids.detector.predict_proba_batch(X, cfg['context_k'])
    pred = np.array(cfg['labels'])[p.argmax(1)]
    present = sorted(set(y))
    print(f'Windows: {len(y)} | accuracy {accuracy_score(y, pred):.4f} | '
          f'macro-F1 {f1_score(y, pred, average="macro", labels=present):.4f}')
    print(classification_report(y, pred, labels=present, digits=3, zero_division=0))

    # 1b) alert-level metrics (3-of-5 smoothing): old argmax rule vs validated threshold
    if 'segment' in df:
        bi = cfg['labels'].index('Benign'); score = 1 - p[:, bi]
        rules = [('argmax', p.argmax(1) != bi)]
        if cfg.get('attack_threshold') is not None:
            rules.append((f"threshold {cfg['attack_threshold']}", score >= cfg['attack_threshold']))
        print(f"Alert metrics ({cfg['smoothing_m']}-of-{cfg['smoothing_n']}, {(y == 'Benign').mean()*100:.0f}% benign windows):")
        for name, fl in rules:
            print(f'  {name:<16}', alert_metrics(y, df['segment'].to_numpy(), fl, cfg['smoothing_m'], cfg['smoothing_n']))

    # 2) streaming latency (1 thread) via full pipeline
    lat = []
    for i, rec in enumerate(df.drop(columns=[c for c in ('label', 'segment') if c in df]).head(3000).to_dict('records')):
        lat.append(ids.process(i, rec).latency_ms)
    lat = np.array(lat[100:])
    out = {'machine': platform.processor() or platform.machine(), 'cpus': os.cpu_count(),
           'latency_ms_median': round(float(np.median(lat)), 3),
           'latency_ms_p95': round(float(np.percentile(lat, 95)), 3),
           'throughput_windows_per_s_1thread': round(1000 / float(lat.mean()), 1),
           'model_size_MB': round(sum((Path(cfg['models_dir']) / f).stat().st_size
                                      for f in ('m_window.json', 'm_context.json')) / 1e6, 2)}
    print(json.dumps(out, indent=1))
    return {'model': model, 'macro_f1': round(f1_score(y, pred, average='macro', labels=present), 4), **out}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default=None)
    args = ap.parse_args()
    models = [args.model] if args.model else available_models(ROOT / 'config.yaml')
    res = [evaluate(m) for m in models]
    if len(res) > 1:
        print('\n===== Comparison =====')
        for r in res:
            print(f"{r['model']:<6} macro-F1 {r['macro_f1']:.4f} | latency {r['latency_ms_median']} ms | "
                  f"{r['throughput_windows_per_s_1thread']} windows/s | {r['model_size_MB']} MB")


if __name__ == '__main__':
    main()
