# =====================================================================
# CICIoMT2024 - v10f: Is IAT a (scaled) capture timestamp?
# Hypothesis (from CIC's own pcap2csv code, Feature_extraction.py):
#   IAT = ts - first_pac_time, with first_pac_time reset to 0 every 20 packets (older code path)
#   -> one packet per block gets IAT = absolute Unix time; the 10/20-row window MEAN then gives
#      IAT ~ UnixTime / k  (k = 10 or 20)  -> e.g. 8.47e7*20 = 1.694e9 = Sep 2023.
# This script reads ONLY the IAT column of every raw train/test CSV, and for every file:
#   - share of windows with a huge IAT (> 1e6 s)
#   - k = round(1.69e9 / median_huge_IAT), decoded date = median_huge_IAT * k
#   - Spearman(decoded time, row order) inside the file (time should move forward)
# CPU only, ~5-10 min (reads 72 CSV files, one column each).  Output: results_v10f/
# =====================================================================
import os, re, glob, datetime as dt, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT = '/content/drive/MyDrive/CICIoMT2024'
except ImportError:
    DATA_ROOT = os.environ.get('DATA_ROOT', r'E:\CICIoMT2024')
OUT_DIR = os.path.join(DATA_ROOT, 'results_v10f'); os.makedirs(OUT_DIR, exist_ok=True)

def label_from_file(path):
    name = re.sub(r'_(train|test)\.pcap\.csv$', '', os.path.basename(path)).replace('TCP_IP-', '')
    return re.sub(r'\d+$', '', name)

rows = []
for split in ('train', 'test'):
    files = sorted(glob.glob(os.path.join(DATA_ROOT, split, '*.csv')))
    print(split, len(files), 'files')
    for f in files:
        iat = pd.read_csv(f, usecols=lambda c: c.strip() == 'IAT').iloc[:, 0].astype('float64').values
        big = iat[iat > 1e6]
        r = {'split': split, 'file': os.path.basename(f), 'label': label_from_file(f), 'rows': len(iat),
             'huge_IAT_%': 100 * len(big) / max(len(iat), 1), 'median_IAT': np.median(iat)}
        if len(big):
            med = np.median(big); k = int(round(1.69e9 / med))
            ts = big * k
            r.update({'median_huge_IAT': med, 'k_window': k,
                      'decoded_start_utc': str(dt.datetime.utcfromtimestamp(np.percentile(ts, 1)))[:19],
                      'decoded_end_utc': str(dt.datetime.utcfromtimestamp(np.percentile(ts, 99)))[:19],
                      'span_hours': (np.percentile(ts, 99) - np.percentile(ts, 1)) / 3600,
                      'spearman_time_vs_row': spearmanr(np.arange(len(big)), big).correlation if len(big) > 10 else np.nan})
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT_DIR, 'iat_decoded_per_file.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
cols = ['split', 'label', 'file', 'huge_IAT_%', 'k_window', 'decoded_start_utc', 'decoded_end_utc', 'span_hours', 'spearman_time_vs_row']
print('\nPer file (sorted by decoded start time):')
print(R.sort_values('decoded_start_utc')[cols].round(3).to_string(index=False))

print('\nSummary:')
print(f"Files with huge IAT values: {(R['huge_IAT_%'] > 0).sum()} / {len(R)}")
print(f"Median share of huge-IAT windows per file: {R['huge_IAT_%'].median():.1f}%")
print('k values found:', R['k_window'].value_counts(dropna=True).to_dict())
ok = R.dropna(subset=['decoded_start_utc'])
print('Overall decoded capture period:', ok['decoded_start_utc'].min(), '->', ok['decoded_end_utc'].max())
# do train and test files of the same class decode to overlapping times? (explains why IAT transfers)
same = []
for lab, g in ok.groupby('label'):
    tr, te = g[g.split == 'train'], g[g.split == 'test']
    if len(tr) and len(te):
        same.append({'label': lab, 'train_start': tr['decoded_start_utc'].min(), 'train_end': tr['decoded_end_utc'].max(),
                     'test_start': te['decoded_start_utc'].min(), 'test_end': te['decoded_end_utc'].max()})
print('\nTrain vs test decoded capture times per class:'); print(pd.DataFrame(same).to_string(index=False))
print('\n==== v10f DONE ==== Paste everything from "Per file" down.')
