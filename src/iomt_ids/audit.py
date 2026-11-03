"""M0 - Dataset artifact audit: detects a capture-timestamp encoded in the IAT column of CIC-style CSVs.

Finding (research log s.27): in CICIoMT2024, IAT x k (k = window size 10 or 20) equals the Unix time of
capture (Aug-Sep 2023) for all 72 files, so IAT identifies the recording session (and therefore the class).
In CICIoT2023 (same lab) IAT holds genuine inter-arrival times - a natural control.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

UNIX_LO, UNIX_HI = 1.42e9, 1.80e9          # 2015 .. 2027
HUGE = 1e6                                  # an inter-arrival time of > 11 days per packet is not physical


def _utc(x):
    return dt.datetime.fromtimestamp(float(x), dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def decode_iat_values(iat):
    """Returns dict with share of huge IAT values, window factor k and decoded capture time range."""
    iat = np.asarray(iat, dtype='float64'); big = iat[iat > HUGE]
    out = {'rows': int(len(iat)), 'median_IAT': float(np.median(iat)) if len(iat) else None,
           'huge_IAT_%': round(100 * len(big) / max(len(iat), 1), 3), 'k_window': None,
           'decoded_start_utc': None, 'decoded_end_utc': None, 'timestamp_like': False}
    if len(big):
        med = float(np.median(big))
        ks = [k for k in (1, 2, 5, 10, 20, 50, 100) if UNIX_LO <= med * k <= UNIX_HI]
        if ks:
            ts = big * ks[0]
            out.update({'k_window': ks[0], 'decoded_start_utc': _utc(np.percentile(ts, 1)),
                        'decoded_end_utc': _utc(np.percentile(ts, 99)), 'timestamp_like': True})
    return out


def decode_iat_file(path, column='IAT'):
    iat = pd.read_csv(path, usecols=lambda c: c.strip().lower() == column.lower()).iloc[:, 0]
    return {'file': Path(path).name, **decode_iat_values(iat)}


def audit_folder(folder, pattern='*.csv'):
    files = sorted(Path(folder).rglob(pattern))
    return pd.DataFrame([decode_iat_file(f) for f in files])
