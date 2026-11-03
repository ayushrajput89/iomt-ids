"""Full IDS pipeline: M1 guard -> M3 context -> M4 detector -> M6 alerts -> M7 explanations."""
import time
from dataclasses import dataclass
from typing import Optional

from .alert_manager import Alert, AlertManager
from .artifact_guard import ArtifactGuard
from .context_buffer import ContextBuffer
from .detector import ContextAwareDetector
from .explainer import Explainer


@dataclass
class Detection:
    index: int
    predicted: str
    confidence: float
    latency_ms: float
    attack_score: float = 0.0          # 1 - P(Benign)
    flagged: bool = False              # window counted as attack by the alert rule
    alert: Optional[Alert] = None


class IDSPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.guard = ArtifactGuard(cfg['features'], cfg['artifact_features'], cfg['redundant_features'])
        self.context = ContextBuffer(cfg['context_k'])
        self.detector = ContextAwareDetector(cfg['models_dir'], cfg['labels'], cfg['fusion_w'], cfg.get('nthreads', 1))
        self.alerts = AlertManager('Benign', cfg['smoothing_m'], cfg['smoothing_n'], cfg.get('severity'),
                                   cfg.get('min_confidence', 0.0))
        self.explainer = Explainer(self.detector.window_model, cfg['features'], cfg['labels'],
                                  context_booster=self.detector.context_model)
        self.benign_idx = cfg['labels'].index('Benign')
        self.threshold = cfg.get('attack_threshold')        # None -> argmax rule

    def set_threshold(self, t):
        self.threshold = t

    def reset(self):
        self.context.reset(); self.alerts.reset()

    def process(self, index, record) -> Detection:
        t0 = time.perf_counter()
        x = self.guard.transform(record)
        ctx = self.context.push(x)
        self.last_x, self.last_ctx = x, ctx                 # kept for on-demand explanations
        p = self.detector.predict_proba_one(x, ctx)
        c = int(p.argmax()); label = self.cfg['labels'][c]; conf = float(p[c])
        score = 1.0 - float(p[self.benign_idx])
        if self.threshold is None:
            flagged, a_label, a_conf = c != self.benign_idx, label, conf
        else:
            q = p.copy(); q[self.benign_idx] = -1.0; a = int(q.argmax())      # most likely ATTACK class
            flagged, a_label, a_conf = score >= self.threshold, self.cfg['labels'][a], float(p[a])
        latency = (time.perf_counter() - t0) * 1000          # detection latency (explanation alag)
        alert = self.alerts.update(index, a_label if flagged else 'Benign', a_conf)
        if alert is not None and alert.event != 'CLEARED':
            alert.reasons = self.explainer.explain(x, alert.attack_type)
        return Detection(index, label, conf, latency, score, bool(flagged), alert)
