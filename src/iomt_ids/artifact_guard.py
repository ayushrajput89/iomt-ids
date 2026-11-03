"""M1 - Artifact Guard: removes the leaked and redundant features found in the audit from the input."""
import numpy as np


class ArtifactGuard:
    def __init__(self, features, artifact_features, redundant_features):
        self.features = list(features)                       # model ka exact feature order
        self.blocked = set(artifact_features) | set(redundant_features)
        leaked = self.blocked & set(self.features)
        if leaked:
            raise ValueError(f'The model uses blocked features: {sorted(leaked)}')

    @staticmethod
    def _norm(name):
        return str(name).strip().replace(' ', '_')

    def transform(self, record):
        """record: dict / pandas Series (raw CIC columns allowed). Returns float32 vector in model order."""
        clean = {self._norm(k): v for k, v in dict(record).items()}
        missing = [f for f in self.features if f not in clean]
        if missing:
            raise KeyError(f'Missing features in the input: {missing[:5]}')
        return np.array([clean[f] for f in self.features], dtype='float32')

    def transform_frame(self, df):
        df = df.rename(columns=self._norm)
        return df[self.features].to_numpy(dtype='float32')

    def report(self, columns):
        cols = {self._norm(c) for c in columns}
        return {'dropped_artifact_or_redundant': sorted(cols & self.blocked),
                'used_features': len(self.features)}
