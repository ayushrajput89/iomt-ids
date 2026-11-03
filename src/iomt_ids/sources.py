"""Traffic sources. Now: parquet/CSV replay (trace-driven). Later: PCAP / live capture."""
import pandas as pd


class ParquetReplaySource:
    """Demo stream (from Colab v9 Cell 8): feature columns + 'label' (+ 'segment')."""
    def __init__(self, path, limit=None):
        self.df = pd.read_parquet(path)
        if limit: self.df = self.df.iloc[:limit]

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        has_label = 'label' in self.df.columns
        for i, row in enumerate(self.df.itertuples(index=False)):
            rec = row._asdict()
            yield i, rec, (rec.pop('label') if has_label else None)


class CSVReplaySource:
    """Raw CICIoMT2024 CSV (45 columns, no label column). The artifact guard removes the extra columns."""
    def __init__(self, path, limit=None, label=None):
        self.df = pd.read_csv(path, nrows=limit); self.label = label

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        for i, (_, row) in enumerate(self.df.iterrows()):
            yield i, row.to_dict(), self.label
