# IoMT-IDS

An intrusion detection system (IDS) for Internet of Medical Things (IoMT) networks, built and tested on the
CICIoMT2024 dataset (19 classes: benign traffic and 18 attacks from the DDoS, DoS, MQTT, reconnaissance and
ARP spoofing families).

This repository contains the code that goes with our paper on timestamp leakage in CICIoMT2024 (see
[Paper](#paper)). It includes the detection pipeline, a Streamlit dashboard, a zero-day lab and the scripts we used
for the experiments.

## Background

Many studies report about 99% accuracy on CICIoMT2024 for the 19-class task. While working with the dataset, we found
that the `IAT` column does not hold the time between packets. Multiplied by the window size, it gives the Unix time at
which each traffic file was recorded. Every attack was recorded in its own session, so this column tells a model which
recording a window comes from, and therefore its class. A model trained on `IAT` alone reaches 98.9% accuracy.

This IDS removes `IAT` and the two window-size columns (`Number`, `Weight`), uses the recent history of the traffic,
and raises alerts with a threshold chosen on validation data.

Main results (official test files, leakage-free features):

| Result | Value |
|---|---|
| Accuracy with `IAT` alone | 98.9% |
| Best other single feature | 63.8% |
| 19-class accuracy without the leaked feature | about 72% |
| Same, with DoS and DDoS of each protocol merged (13 classes) | 99.3% |
| Context model vs per-window model on mixed traffic | +4.5 macro-F1 (5 seeds, p = 0.0004) |
| False alarms at 95% benign traffic | 5.8 to 0.37 per 1000 benign windows |
| Attack segments still detected | 96.5% |
| Accuracy of a model that knows only the window size | 24.8% (so window size is not a second leak) |

## How it works

Each traffic window goes through these steps:

1. **Artifact guard.** Removes `IAT`, `Number`, `Weight` and three redundant columns (`Drate`, `Srate`, `LLC`).
   39 features remain.
2. **Context buffer.** Keeps the last 10 windows and adds their mean and standard deviation.
3. **Detector.** Two XGBoost models, one on the current window and one on the context, are combined with a weight.
4. **Alert manager.** A window counts as an attack when its attack probability is at least 0.95 (Lite model). An
   alert is raised when 3 of the last 5 windows are attacks. Alerts are raised, updated when the attack type changes,
   and cleared.
5. **Explainer.** Each alert shows its top 3 reasons, computed with SHAP.

The zero-day lab (`app/zero_day_lab.py`) adds three more parts. A novelty detector flags windows that look unlike any
known class, a pattern step groups these windows so an analyst can name them, and a continual-learning step teaches
the model the new class while keeping the old ones (using a replay buffer).

## Project layout

```
iomt-ids/
  config.yaml                settings (context size, alert rule, severity, blocked features)
  models/lite/               Lite model, the default (22 MB, about 2 ms per window)
  data/demo_stream.parquet   20,000 windows of mixed test traffic for the demo
  data/zeroday/              reference data and results for the zero-day lab
  src/iomt_ids/              the IDS code (guard, context buffer, detector, alerts, SHAP, zero-day modules)
  app/dashboard.py           Streamlit dashboard
  app/zero_day_lab.py        Streamlit zero-day lab
  scripts/                   terminal demo, evaluation, dataset audit, HTML report, zero-day lab
  tests/                     unit tests
  notebooks/                 Colab scripts for training and the paper experiments (v9 to v14)
```

## Setup

Tested on Windows with Python 3.11.

```bash
git clone https://github.com/ayushrajput89/iomt-ids.git
cd iomt-ids
python -m venv .venv
.venv\Scripts\activate          # on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The Lite model and the demo stream are included, so everything below works after installation. The Full model
(209 MB) is too large for GitHub. It can be trained with `notebooks/v9_ids_system.py` and placed in `models/`.
The model in use is set by `active_model` in `config.yaml`.

## Usage

```bash
python -m pytest tests -q                        # 13 tests
python scripts/replay_demo.py --limit 3000       # replay traffic in the terminal and print alerts
python scripts/evaluate.py                       # macro-F1, alert metrics, latency and size of each model
python scripts/audit_dataset.py <csv-folder>     # check a CIC-style dataset for the IAT timestamp problem
python scripts/make_report.py                    # HTML report with metrics, plots and SHAP
streamlit run app/dashboard.py                   # dashboard
streamlit run app/zero_day_lab.py                # zero-day lab
python scripts/zero_day_lab.py                   # zero-day lab for all 8 scenarios, from the terminal
```

The dashboard has seven tabs: live monitor, performance, explanations, model comparison, the leakage analysis,
research results, and system details.

## Models

Lite, Full and Tiny models on mixed test streams (Colab CPU, one thread):

| Model | Depth / trees | Macro-F1 | Latency per window | Size |
|---|---|---|---|---|
| Lite (default) | 6 / 150 | 72.6 | 2.0 ms | 22 MB |
| Full | 8 / 1000 | 72.4 | 18.2 ms | 209 MB |
| Tiny | 4 / 80 | 71.0 | 1.3 ms | 5 MB |

Comparison of twelve models on the same 39 features (macro-F1, %):

| Model | Mixed streams | Official test set |
|---|---|---|
| Our context ensemble | 70.2 | - |
| LightGBM / XGBoost with context | 69.8 | 68.9 / 68.7 |
| Random forest | 68.0 | 67.2 |
| Transformer (last 10 windows) | 60.5 | - |
| GRU (last 10 windows) | 59.5 | - |
| MLP / 1D-CNN | about 57 | 55.9 / 55.2 |
| Logistic regression | 52.8 | 50.6 |

More details on the data, training and limitations are in `MODEL_CARD.md`.

## Zero-day lab

To test unseen attacks, we remove one attack group from training at a time (8 scenarios) and check three things:
whether the new attack is noticed, whether its windows can be grouped for an analyst, and whether the model can learn
it without forgetting the old classes. Results on the official test set, averaged over the 8 scenarios:

- The novelty score (kNN distance combined with an energy score) separates unseen from known traffic with an AUROC
  of 94.2%. About 89% of unseen-attack windows are either flagged as unknown or alerted as some known attack.
  Low-rate attacks such as malformed MQTT messages are harder (52%).
- Learning a new class with a replay buffer keeps old-class macro-F1 at 66.0% (65.6% before) and reaches 66.2% on the
  new class. Without the replay buffer, old-class macro-F1 drops to 26.5%.

## Limitations

- All results come from replaying recorded traffic of one testbed. The system has not been tested on a live
  hospital network.
- The input is CIC window features. A feature extractor for raw packet captures is not included.
- From a single window, DoS and DDoS attacks of the same protocol cannot be told apart, and some low-rate attacks
  (ARP spoofing, vulnerability scans) remain hard.

## Dataset

S. Dadkhah et al., "CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT,"
Internet of Things, vol. 28, 101351, 2024. The CSV files are not included here. They can be downloaded from the
Canadian Institute for Cybersecurity and have their own terms of use.

## Paper

A. Rajput, N. Jain and R. Amin, "Revisiting Intrusion Detection on CICIoMT2024: Timestamp Leakage and
Context-Aware Detection," manuscript in preparation, 2026.

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT License. See `LICENSE`.
