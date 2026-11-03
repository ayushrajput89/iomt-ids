"""Alert-level metrics (same definitions as the research notebooks v9/v10d/v10e)."""
import numpy as np
import pandas as pd


def smooth(flags, m, n):
    """m-of-n: alarm is ON when at least m of the last n windows are flagged."""
    return pd.Series(np.asarray(flags, dtype=int)).rolling(n, min_periods=1).sum().to_numpy() >= m


def alert_metrics(y_true, segments, flags, m=3, n=5, benign='Benign'):
    """y_true: labels per window, segments: segment id per window, flags: bool attack flag per window."""
    y_true, segments, flags = np.asarray(y_true), np.asarray(segments), np.asarray(flags, dtype=bool)
    on = smooth(flags, m, n); ben = y_true == benign
    d = np.diff(np.r_[0, on.astype(int), 0]); starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    events_true = [(~ben[s:e]).any() for s, e in zip(starts, ends)]
    seg_starts = np.r_[0, np.flatnonzero(np.diff(segments)) + 1]; seg_ends = np.r_[seg_starts[1:], len(segments)]
    det, delay = [], []
    for s, e in zip(seg_starts, seg_ends):
        if y_true[s] == benign: continue
        hit = np.flatnonzero(smooth(flags[s:e], m, n)); det.append(len(hit) > 0)
        if len(hit): delay.append(int(hit[0]))
    return {'false_alarms_per_1000_benign': round(float(sum(ben[s] for s in starts)) / max(int(ben.sum()), 1) * 1000, 2),
            'alert_events': len(starts),
            'event_precision_%': round(float(np.mean(events_true)) * 100, 2) if len(starts) else None,
            'window_precision_%': round(float((~ben[on]).mean()) * 100, 2) if on.any() else None,
            'attack_segments_detected_%': round(float(np.mean(det)) * 100, 2) if det else None,
            'median_delay_windows': float(np.median(delay)) if delay else None}
