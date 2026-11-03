"""Terminal demo: replay the demo stream window by window and watch live alerts.

Usage (from the project folder):
    python scripts/replay_demo.py                 # poora demo stream
    python scripts/replay_demo.py --limit 3000 --speed 200   # 200 windows/sec pe
    python scripts/replay_demo.py --model lite --limit 3000  # Lite model
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from iomt_ids import IDSPipeline, ParquetReplaySource, load_config   # noqa: E402

COLORS = {'HIGH': '\033[91m', 'MEDIUM': '\033[93m', 'CLEARED': '\033[92m', 'END': '\033[0m'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'config.yaml'))
    ap.add_argument('--model', default=None, help='full | lite (default: config active_model)')
    ap.add_argument('--data', default=None, help='parquet stream (default: config demo_stream)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--speed', type=float, default=0, help='windows per second (0 = jitna fast ho sake)')
    args = ap.parse_args()

    cfg = load_config(args.config, args.model)
    ids = IDSPipeline(cfg)
    src = ParquetReplaySource(args.data or cfg['demo_stream'], args.limit)
    print(f'IoMT-IDS [{cfg["active_model"]}] | {len(cfg["features"])} features | K={cfg["context_k"]} | w={cfg["fusion_w"]} | '
          f'smoothing {cfg["smoothing_m"]}-of-{cfg["smoothing_n"]} | windows: {len(src)}\n')

    n_alerts, lat, correct, total = 0, [], 0, 0
    for i, rec, true in src:
        d = ids.process(i, rec)
        lat.append(d.latency_ms)
        if true is not None:
            total += 1; correct += d.predicted == true
        a = d.alert
        if a:
            if a.event == 'CLEARED':
                print(f"{COLORS['CLEARED']}[{i:>6}] CLEARED   {a.attack_type}{COLORS['END']}")
            else:
                n_alerts += a.event == 'RAISED'
                c = COLORS.get(a.severity, '')
                truth = f' | true={true}' if true is not None else ''
                print(f"{c}[{i:>6}] {a.event:<12} {a.severity:<6} {a.attack_type:<24} conf={a.confidence:.2f} "
                      f"flagged={a.windows_flagged}{truth}{COLORS['END']}\n          why: {', '.join(a.reasons)}")
        if args.speed:
            time.sleep(1 / args.speed)

    lat = sorted(lat)
    print(f'\nWindows: {len(lat)} | alerts raised: {n_alerts} | '
          f'latency median {lat[len(lat)//2]:.2f} ms, p95 {lat[int(len(lat)*.95)]:.2f} ms'
          + (f' | window accuracy {correct/total*100:.2f}%' if total else ''))


if __name__ == '__main__':
    main()
