"""M3 - Context Buffer: pichhli K windows ka rolling mean + std (pandas rolling(min_periods=1), ddof=1 jaisa)."""
from collections import deque

import numpy as np


class ContextBuffer:
    def __init__(self, k=10):
        self.k = k
        self.buf = deque(maxlen=k)

    def reset(self):
        self.buf.clear()

    def push(self, x):
        """Add current window and return [x, mean, std] context vector."""
        self.buf.append(np.asarray(x, dtype='float32'))
        b = np.asarray(self.buf)
        std = b.std(0, ddof=1) if len(b) > 1 else np.zeros_like(x, dtype='float32')
        return np.hstack([x, b.mean(0), std]).astype('float32')

    def __len__(self):
        return len(self.buf)
