# =====================================================================
# v11a - CICIoT2023 quick check: does IAT also encode the capture timestamp?
# Reads ONLY the IAT column of every CSV in MyDrive/CICIoT2023/ (one file per pcap recording).
# For each file: share of huge IAT values, best window factor k (IAT*k lands in a 2015-2027
# Unix-time range), decoded capture start/end. CPU only, ~2-5 min.
# =====================================================================
import os, re, glob, datetime as dt, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/CICIoT2023'
except ImportError:
    ROOT = os.environ.get('ROOT2023', r'E:\CICIoT2023')

LO, HI = 1.42e9, 1.80e9                       # 2015 .. 2027 in Unix seconds
def utc(x): return str(dt.datetime.utcfromtimestamp(x))[:19]

rows = []
files = sorted(glob.glob(os.path.join(ROOT, '*.csv')))
print('files found:', len(files))
for f in files:
    iat = pd.read_csv(f, usecols=lambda c: c.strip().lower() == 'iat').iloc[:, 0].astype('float64').values
    big = iat[iat > 1e6]
    r = {'file': os.path.basename(f), 'label': re.sub(r'\d*\.pcap\.csv$', '', os.path.basename(f)),
         'rows': len(iat), 'median_IAT': np.median(iat), 'huge_IAT_%': 100 * len(big) / max(len(iat), 1)}
    if len(big):
        med = np.median(big)
        ks = [k for k in (1, 2, 5, 10, 20, 50, 100) if LO <= med * k <= HI]
        k = ks[0] if ks else None
        r.update({'median_huge_IAT': med, 'k_window': k})
        if k:
            ts = big * k
            r.update({'decoded_start_utc': utc(np.percentile(ts, 1)), 'decoded_end_utc': utc(np.percentile(ts, 99)),
                      'span_hours': (np.percentile(ts, 99) - np.percentile(ts, 1)) / 3600})
    rows.append(r)
R = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, 'results_v11a'), exist_ok=True)
R.to_csv(os.path.join(ROOT, 'results_v11a', 'iat_decoded_2023.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda v: f'{v:,.6g}')
print(R.sort_values('decoded_start_utc', na_position='last').to_string(index=False))
print('\nIAT quantiles per file (to see the raw scale even if nothing decodes):')
for f in files:
    iat = pd.read_csv(f, usecols=lambda c: c.strip().lower() == 'iat').iloc[:, 0].astype('float64')
    print(f'{os.path.basename(f):40s}', np.round(iat.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).values, 4))
print('\n==== v11a DONE ==== paste everything above')
