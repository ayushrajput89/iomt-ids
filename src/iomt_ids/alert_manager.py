"""M6 - Alert Manager: m-of-n smoothing (alarm fatigue kam), severity, alarm start/clear events."""
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Alert:
    event: str                    # 'RAISED' | 'TYPE_CHANGED' | 'CLEARED'
    window_index: int
    attack_type: str
    severity: str
    confidence: float
    windows_flagged: str          # e.g. '4/5'
    reasons: List[str] = field(default_factory=list)


class AlertManager:
    def __init__(self, benign_label='Benign', m=3, n=5, severity_map=None, min_confidence=0.0):
        self.benign, self.m, self.n = benign_label, m, n
        self.sev = severity_map or {'default': 'MEDIUM'}
        self.min_conf = min_confidence
        self.recent = deque(maxlen=n)          # (label, confidence)
        self.active: Optional[str] = None

    def reset(self):
        self.recent.clear(); self.active = None

    def severity(self, label):
        for key, level in self.sev.items():
            if key != 'default' and (label == key or label.startswith(key + '-') or label.startswith(key)):
                return level
        return self.sev.get('default', 'MEDIUM')

    def update(self, idx, label, conf) -> Optional[Alert]:
        is_attack = label != self.benign and conf >= self.min_conf
        self.recent.append((label if is_attack else self.benign, conf))
        attacks = [(l, c) for l, c in self.recent if l != self.benign]
        flagged = f'{len(attacks)}/{len(self.recent)}'
        if len(attacks) >= self.m:
            top = Counter(l for l, _ in attacks).most_common(1)[0][0]
            mean_conf = sum(c for l, c in attacks if l == top) / sum(1 for l, _ in attacks if l == top)
            if self.active is None:
                self.active = top
                return Alert('RAISED', idx, top, self.severity(top), round(mean_conf, 3), flagged)
            if top != self.active:
                self.active = top
                return Alert('TYPE_CHANGED', idx, top, self.severity(top), round(mean_conf, 3), flagged)
            return None
        if self.active is not None:
            prev, self.active = self.active, None
            return Alert('CLEARED', idx, prev, self.severity(prev), 0.0, flagged)
        return None
