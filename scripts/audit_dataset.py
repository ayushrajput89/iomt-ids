"""Scan a folder of CIC-style CSV files for a timestamp hidden in the IAT feature.

Usage:  python scripts/audit_dataset.py E:\\CICIoMT2024\\train
        python scripts/audit_dataset.py E:\\CICIoMT2024 --out audit.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from iomt_ids import audit_folder   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='Detect capture-timestamp leakage in the IAT column.')
    ap.add_argument('folder'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    df = audit_folder(a.folder)
    if df.empty:
        print('No CSV files found.'); return
    cols = ['file', 'rows', 'median_IAT', 'huge_IAT_%', 'k_window', 'decoded_start_utc', 'decoded_end_utc']
    print(df[cols].sort_values('decoded_start_utc', na_position='last').to_string(index=False))
    bad = df['timestamp_like'].sum()
    print(f'\nFiles scanned: {len(df)} | files with timestamp-like IAT: {bad}')
    if bad:
        print(f"Decoded capture period: {df['decoded_start_utc'].dropna().min()} -> {df['decoded_end_utc'].dropna().max()} (UTC)")
        print('VERDICT: IAT encodes WHEN a file was captured, not attack behaviour. Remove it before training.')
    else:
        print('VERDICT: IAT looks like a genuine inter-arrival time in every file.')
    if a.out:
        df.to_csv(a.out, index=False); print('saved', a.out)


if __name__ == '__main__':
    main()
