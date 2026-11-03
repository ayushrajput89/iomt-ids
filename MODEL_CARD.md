# Model card — IoMT-IDS (leakage-aware, context-aware XGBoost ensemble)

| | |
|---|---|
| **Task** | 19-class network intrusion detection for Internet of Medical Things (Benign + 18 attacks: DDoS, DoS, MQTT, Recon, ARP spoofing) |
| **Data** | CICIoMT2024 (Dadkhah et al., *Internet of Things* 28, 2024), WiFi + MQTT CSVs, official train/test split |
| **Input** | One CIC traffic window (45 columns) -> 39 features after the artifact guard |
| **Removed features** | `IAT` (encodes the capture Unix timestamp / window size; alone gives 98.9% accuracy), `Number`, `Weight` (window bookkeeping), `Drate`, `Srate`, `LLC` (constant / duplicate) |
| **Model** | p = w * P(XGBoost on current window) + (1 - w) * P(XGBoost on window + rolling mean/std of last 10 windows) |
| **Variants** | lite (default, depth 6, 150 trees, 22 MB), full (depth 8, 1000 trees, 209 MB), tiny (depth 4, 80 trees, 5 MB) |
| **Alert rule** | window flagged if P(attack) >= 0.95 (lite) / 0.98 (full), chosen on validation streams; alert when 3 of the last 5 windows are flagged |
| **Explanations** | Exact TreeSHAP: top-3 reasons per alert, full waterfall per alert, global and per-class importance |

## Training (done in the research notebooks, Google Colab)
- `notebooks/v9_ids_system.py` (full) and `notebooks/v9b_ids_lite.py` (lite / tiny): XGBoost `hist`, balanced sample weights,
  early stopping on a 15% split; training data = mixed benign/attack streams built from train blocks (segment lengths 5-1000);
  fusion weight w tuned on held-out train blocks.
- Model comparison: `notebooks/v12_model_zoo.py` (12 models).

## Results (official test data, leakage-free)
| Setting | Accuracy | Macro-F1 |
|---|---|---|
| Static official test, XGBoost | 72.0% | 68.7 |
| Mixed hospital-like stream, our ensemble (v12) | 83.6% | 70.2 |
| Same, best deep model (Transformer over 10 windows) | 73.8% | 60.5 |
| 13 classes (DoS+DDoS of same protocol merged) | 99.3% | 82.1 |

Alerting at 95% benign traffic (lite, threshold 0.95): 0.37 false alarms per 1000 benign windows, 72% of alerts are real attacks,
96.5% of attack segments detected, median delay 2 windows. Latency ~2 ms (Colab CPU) / ~6.7 ms (laptop, full pipeline), 1 thread.

Window size is not a second leak (v14): a classifier that knows only the flood / low-rate window-size group reaches
24.8% accuracy (5.7 macro-F1); only 0.26% of the model's errors cross the two groups.

## Known limitations
- Trace-driven replay of one testbed dataset; not validated on a live hospital network.
- DoS vs DDoS of the same protocol is weakly identifiable from window features (97.6% of static errors).
- Low-rate attacks are hard: ARP spoofing F1 ~34, Recon-VulScan ~37; most benign false alarms are predicted as ARP spoofing.
- Rows inside a CIC CSV are only time-ordered within 10 MB pcap chunks, which limits the realism of context features.
- Single training seed for the 12-model comparison.

## Zero-day lab (Phase 2)
- Lab model: window-level XGBoost (depth 6, ≤150 trees) trained on the exported reference set (1000 windows/class) without the
  hidden attack group; 19 label slots are kept so a newly named class can be fine-tuned in.
- Unknown detection: kNN distance (k = 10, signed-log z-space) + energy score; threshold = 95th percentile on known validation
  windows (v13: mean AUROC 94.2, 4.8% false-unknown on the known test set); flagged only when sustained 3-of-5 windows.
- Continual learning: continued boosting on new windows + replay buffer (500/class); every update is versioned in
  `models/lab/registry.json`. v13: new-class F1 66.2 with no forgetting vs 26.5 old-class F1 without replay.
- Evaluation: the demo stream is split by whole segments into a discovery half (analyst labels come from here) and an
  evaluation half (never labelled). Research numbers on the official test set: v13 (Z1–Z4).
- Limits: low-rate unseen attacks (ARP spoofing, malformed MQTT) are often not separable from normal traffic; the analyst is
  simulated (pattern labels may be impure); window-level model only; thresholds assume the known
  traffic mix of the reference set.

## Intended use
Research and teaching demo of honest IDS evaluation (leakage audit, mixed-stream testing, base-rate-aware alerting). Not a certified medical-device security product.
