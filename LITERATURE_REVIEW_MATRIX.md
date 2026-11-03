# Literature Review Matrix — Shortcut-aware, Context-aware IoMT IDS (draft, 9 Oct 2026)

How this was built: web search across 7 themes, with most entries confirmed on publisher, arXiv or DBLP pages.

**Verify before citing.** Each row must be checked against its actual PDF.
- `[V?]` means a detail such as the DOI or volume was not confirmed.
- ⭐ marks a must-read: one of the papers closest to our work.
- The CICIoMT2024-specific audit of 20 papers is kept separately in `LITERATURE_AUDIT_CICIoMT2024.md`.

Suggested order for Section II (Related Work): A → B → C → D → E → F → G, then a gap statement.

---

## A. IoMT security and IDS surveys (background)
| Ref | Year | Venue | DOI / URL | Key point | Use in our paper |
|---|---|---|---|---|---|
| Sun et al., "Security and Privacy for the IoMT Enabled Healthcare Systems: A Survey" | 2019 | IEEE Access 7 | 10.1109/ACCESS.2019.2960617 | IoMT threats; devices are resource-limited | Motivates a lightweight IDS |
| Yaacoub et al., "Securing IoMT systems: Limitations, issues and recommendations" | 2020 | FGCS | 10.1016/j.future.2019.12.028 | Crypto alone is not enough; argues for IDS | IDS as a needed layer |
| Ghubaish et al., "Recent Advances in the IoMT Systems Security" | 2021 | IEEE IoT Journal | 10.1109/JIOT.2020.3045653 | Security across data collection, transmission and storage | Threat landscape anchor |
| Si-Ahmed et al., "Survey of ML Based IDS Methods for IoMT" | 2022 | arXiv 2202.09657 [journal V?] | 10.48550/arXiv.2202.09657 | Layer-wise survey of ML IDS | Place our work at the network layer |
| Siqueira et al., "IDS for Healthcare 5.0 survey" | 2025 | Sensors 25(20):6261 | 10.3390/s25206261 | Compares datasets; calls for XAI; CICIoMT2024 only partly aligned | Motivates XAI |
| ⭐ Fatima et al., "IDS for IoMT in smart hospitals: SLR" | 2026 | Internet of Things 39, 102008 | 10.1016/j.iot.2026.102008 | 32 studies report 98–99.9% accuracy; XGBoost most common; resource limits and real-time use ignored; **calls for context-aware IDS** | Strong support for our critique and design |

## B. IoT / IoMT IDS datasets and dataset critique
| Ref | Year | Venue | DOI / URL | Key point | Use |
|---|---|---|---|---|---|
| Meidan et al., N-BaIoT | 2018 | IEEE Pervasive Computing | 10.1109/MPRV.2018.03367731 [V?] | Botnet traffic from 9 devices; autoencoders; **majority vote over a sliding window** | Dataset history + prior alert smoothing (see E) |
| Koroniotis et al., Bot-IoT | 2019 | FGCS 100 | arXiv 1811.00701 | Simulated botnet testbed | Testbed artifact risk |
| Ullah & Mahmoud, IoTID20 | 2020 | Canadian AI, LNCS | dblp conf/ai/UllahM20 | Flow-feature smart-home dataset | — |
| Alsaedi et al., TON_IoT | 2020 | IEEE Access 8 | 10.1109/ACCESS.2020.3022862 | Telemetry + logs + network data | — |
| Hady et al., WUSTL-EHMS-2020 | 2020 | IEEE Access 8 | 10.1109/ACCESS.2020.3000421 | ~16k records; network + biometric features | Older, small IoMT dataset → why we chose CICIoMT2024 |
| Ahmed et al., ECU-IoHT | 2021 | Ad Hoc Networks 122 | 10.1016/j.adhoc.2021.102621 | Healthcare testbed; few attacks | — |
| Ferrag et al., Edge-IIoTset | 2022 | IEEE Access 10 | 10.1109/ACCESS.2022.3165809 | 14 attacks; centralized + federated | — |
| ⭐ Neto et al., CICIoT2023 | 2023 | Sensors 23(13):5941 | 10.3390/s23135941 | Same CIC pipeline as ours; **RF accuracy 99.16% but F1 0.714 at 34 classes** | Accuracy vs F1 gap; sister dataset |
| Dadkhah et al., CICIoMT2024 (base paper) | 2024 | Internet of Things 28 | CIC page / PII S2542660524002920 [DOI V?] | 40 devices; WiFi/MQTT/BLE; 18 attacks | Our dataset |
| Sarhan et al., "Standard feature set for NIDS datasets" | 2022 | MONET | 10.1007/s11036-021-01843-0 | Feature design drives results | Supports a feature audit |
| Apejoye et al., "Critical review of NIDS benchmark datasets for IoT" | 2025 | Comp. Networks & Comm. 3(2) | 10.37256/cnc.3220257228 | Inconsistent testbeds and features → poor generalization | Dataset critique |
| ⭐ Kostas et al., IoTGeM | 2025 | Computer Networks | 10.1016/j.comnet.2025.111550 | Finds "common data leaks" in IoT IDS work; strict separation; window features; SHAP | **Close precedent** for leakage-aware evaluation |

## C. ML/DL IDS on IoT/IoMT (beyond CICIoMT2024)
| Ref | Year | Venue | DOI | Dataset → result | Use |
|---|---|---|---|---|---|
| Thamilarasu et al., "An IDS for IoMT" | 2020 | IEEE Access 8 | 10.1109/ACCESS.2020.3026260 | Simulated WBAN; resource-aware | Early IoMT IDS |
| Alani & Miri, "Explainable universal feature set for IoT IDS" | 2022 | Sensors 22(15):5690 | 10.3390/s22155690 | 6 features + SHAP → 99.6% | Tiny feature set + SHAP; its features (dst_port, proto) may be shortcuts |
| Vijayakumar et al., DNN for IoHT | 2023 | Processes 11(4):1072 | 10.3390/pr11041072 | ECU-IoHT accuracy 99.85% but F1 90.3% | Accuracy vs F1 gap |
| Alsolami et al., ensembles for IoMT | 2024 | Sensors 24(18):5937 | 10.3390/s24185937 | WUSTL accuracy 98.88%; random split + oversampling | Typical evaluation weaknesses |
| Shaikh et al., RCLNet | 2024 | Frontiers Digital Health 6 | 10.3389/fdgth.2024.1467241 | WUSTL 99.78%; CNN-LSTM-attention | ≥99% trend |
| Hosain & Cakmak, XAI-XGBoost | 2025 | Scientific Reports 15 | 10.1038/s41598-025-07790-0 | WUSTL 99.22%; XGBoost + SHAP/LIME | Same family as ours; SHAP not used to find shortcuts |
| Wahab et al., DL + transformers on CICIoT2023 | 2025 | Sensors 25(15):4845 | 10.3390/s25154845 | 12-class ~93%; transformer no better than DNN | Finer classes = harder; complexity doesn't help |
| Fahim-Ul-Islam et al., FedIoMT | 2025 | IEEE Trans. Consumer Electronics 71(2) | 10.1109/TCE.2025.3544885 | 99.2–99.38%; IID only | Federated trend |
| Saleh et al., CNN-transformer + XAI | 2026 | IJMLC 17(5) | 10.1007/s13042-026-03055-y | Edge-IIoTset weighted F1 0.82 (= CNN) | Complexity no F1 gain → supports a lightweight model |
| Akar et al., L2D2 LSTM | 2025 | IEEE Access 13 | 10.1109/ACCESS.2025.3526883 | CICIoMT2024 19-class ~98% | Sequence baseline on our dataset |
| + 20 CICIoMT2024 papers | — | — | see LITERATURE_AUDIT file | 13/20 ≥99%; only 1 uses the official split | Audit table |

## D. ⭐ Evaluation pitfalls, leakage, dataset flaws, shortcut learning (most important)
| Ref | Year | Venue | DOI / URL | Key finding | Positioning |
|---|---|---|---|---|---|
| Sommer & Paxson, "Outside the Closed World" | 2010 | IEEE S&P | 10.1109/SP.2010.25 | Lab ML-NIDS results don't transfer | Founding critique |
| ⭐ Arp et al., "Dos and Don'ts of ML in Computer Security" | 2022 | USENIX Security | usenix.org/.../arp; arXiv 2010.09470 | 10 pitfalls incl. spurious correlations, base rate, lab-only evaluation | IAT/Number/Weight = their "spurious correlation" pitfall |
| ⭐ Geirhos et al., "Shortcut learning in deep neural networks" | 2020 | Nature MI 2 | 10.1038/s42256-020-00257-z | Shortcut concept | Our terminology |
| Kapoor & Narayanan, "Leakage and the reproducibility crisis" | 2023 | Patterns 4(9) | 10.1016/j.patter.2023.100804 | Taxonomy of 8 leakage types | Classify IAT as leakage |
| Engelen et al., "Troubleshooting CICIDS2017" | 2021 | IEEE SPW (DLS) | 10.1109/SPW53761.2021.00009 | CICFlowMeter artifacts + label errors | CIC datasets have construction artifacts |
| Lanvin et al., "Errors in CICIDS2017" | 2022 | CRiSIS, LNCS | hal-03775466 [DOI V?] | Fixing errors changes results | Flaws drive numbers |
| Liu et al., "Error prevalence in NIDS datasets" | 2022 | IEEE CNS | 10.1109/CNS56114.2022.9947235 | Errors across CIC-IDS-2017/2018 | Justifies auditing CIC data |
| ⭐ Flood et al., "Bad Design Smells in Benchmark NIDS Datasets" | 2024 | IEEE EuroS&P | 10.1109/EuroSP60621.2024.00042 | Catalogue of dataset "smells" | Closest framework; our artifacts are such smells |
| Kenyon et al., "Are public intrusion datasets fit for purpose" | 2020 | Computers & Security | 10.1016/j.cose.2020.102022 [V?] | Public datasets have limits | Motivation |
| D'hooge et al., inter-dataset generalization | 2020 | JISA | 10.1016/j.jisa.2020.102564 [V?] | CIC-IDS2017 ↔ 2018 transfer fails | Within-dataset 99% is brittle |
| ⭐ D'hooge et al., "Contaminating effect of metadata features" | 2022 | DIMVA, LNCS 13358 | 10.1007/978-3-031-09484-2_2 | **dst_port alone gives 70–100% accuracy in 10 datasets** | Closest "single feature" result; ours look like normal statistics, not metadata |
| ⭐ Jacobs et al., "AI/ML for Network Security: The Emperor has no Clothes" (Trustee) | 2022 | ACM CCS | 10.1145/3548606.3560609 | Shortcuts in CIC-IDS2017 (TTL bits), found by manual tree inspection | Closest prior shortcut work; ours is SHAP-quantified |
| ⭐ Zhao et al., "ShortcutCatcher" | 2026 | ACM PACMNET (CoNEXT) | 10.1145/3808671 | XAI + check dataset find capture-artifact shortcuts in traffic classification | **Same idea, different domain** → we must cite and differentiate |
| ⭐ Roy & Minu, "Unmasking shortcut learning in IoT IDS" | 2026 | arXiv 2609.28725 (preprint) | arxiv.org/abs/2609.28725 | Timestamps inflate macro-F1 92.6 → 99.3; random splits inflate recall | Concurrent work, different dataset; supports IAT-as-timestamp |
| ⭐ Limeira et al., "Beyond Aggregate Accuracy" | 2026 | SBSeg | 10.5753/sbseg.2026.27104 | CICIoMT2024: aggregate accuracy hides recon/spoofing failures; found "no obvious artifacts" | **Closest on our dataset**; we show the artifacts they missed |
| Tolay, "Beyond Detection Accuracy" | 2026 | arXiv 2608.10349 | — | Cost and stability of explanations on CICIoT2023 | Marginal |

## E. Temporal / context / sequence IDS and streaming evaluation
| Ref | Year | Venue | DOI / URL | Mechanism | Positioning |
|---|---|---|---|---|---|
| ⭐ Mirsky et al., Kitsune | 2018 | NDSS | arXiv 1802.09089 | Damped incremental statistics over 5 time windows; autoencoder ensemble | Standard source of context features; unsupervised; one attack per capture |
| Meidan et al., N-BaIoT | 2018 | IEEE Pervasive Comp. | arXiv 1805.03409 | **Majority vote over ws consecutive windows** | Our m-of-n smoothing is not new; we cite this and evaluate it under interleaving |
| Zhong et al., HELAD | 2020 | Computer Networks 169 | 10.1016/j.comnet.2019.107049 | AE + LSTM score fusion; LSTM helps on continuous attacks | Conceptually close per-instance + temporal fusion |
| ⭐ Corsini et al., "Evaluation of sequential ML for NIDS" | 2021 | ARES (ACM) | 10.1145/3465481.3470065 | Sequence-model gain depends on the dataset | Supports "context gain depends on evaluation design" |
| Andresini et al., INSOMNIA | 2021 | AISec (CCS workshop) | 10.1145/3474369.3486864 | Concept drift; temporal split | Complementary (drift vs short-horizon context) |
| Manocchio et al., FlowTransformer | 2024 | Expert Systems w/ Applications | 10.1016/j.eswa.2023.122564 | Transformer over flow sequences | Heavy sequence models vs our cheap rolling statistics |
| Akar et al., L2D2 | 2025 | IEEE Access 13 | 10.1109/ACCESS.2025.3526883 | LSTM on CICIoMT2024 | Evaluated on per-file splits, not mixed streams |
| Gap note | — | — | — | No paper found that tests on randomly interleaved benign/attack segments | Our mixed-stream protocol |

## F. Explainable AI for IDS
| Ref | Year | Venue | DOI | Key point | Positioning |
|---|---|---|---|---|---|
| Neupane et al., X-IDS survey | 2022 | IEEE Access 10 | 10.1109/ACCESS.2022.3216617 | Taxonomy of explainable IDS | Anchor survey |
| Wang et al., "An explainable ML framework for IDS" | 2020 | IEEE Access 8 | 10.1109/ACCESS.2020.2988359 | First widely cited SHAP-for-IDS paper | Precedent; SHAP used only for interpretation |
| Warnecke et al., "Evaluating explanation methods for DL in security" | 2020 | IEEE EuroS&P | arXiv 1906.02108 | Criteria for judging explanations | Justify TreeSHAP |
| Arreche et al., E-XAI | 2024 | IEEE Access 12 | 10.1109/ACCESS.2024.3365140 | Benchmarks SHAP/LIME on NIDS | XAI quality |
| ⭐ Lapuschkin et al., "Unmasking Clever Hans predictors" | 2019 | Nature Comms 10 | 10.1038/s41467-019-08987-4 | Attribution reveals spurious cues | Origin of "XAI as shortcut detector" |
| Krishna et al., "The disagreement problem" | 2024 | TMLR | arXiv 2202.01602 | Explainers disagree | Why exact TreeSHAP; our stability check |
| Lundberg et al., TreeSHAP | 2020 | Nature MI 2 | 10.1038/s42256-019-0138-9 | Exact Shapley values for trees | Our method |
| D'Ambrosio et al., trustworthy IoMT NIDS | 2026 | IFIP Networking [V?] | dl.ifip.org PDF | SHAP on CICIoMT2024 CNN/LSTM | Same dataset; SHAP for compression, not shortcuts |

## G. Deployment: base rate, alert fatigue, lightweight / edge IDS
| Ref | Year | Venue | DOI / URL | Key point | Positioning |
|---|---|---|---|---|---|
| ⭐ Axelsson, "The base-rate fallacy and the difficulty of intrusion detection" | 1999/2000 | ACM CCS / TISSEC 3(3) | 10.1145/357830.357849 [V?] | False alarm rate is the limiting factor | Motivates smoothing + our 95%-benign test (v10c) |
| ⭐ Alahmadi et al., "99% False Positives" | 2022 | USENIX Security | usenix.org/.../alahmadi | SOC analysts want reliable, explainable, contextual alerts | Our alerts aim at these properties |
| Hassan et al., NoDoze | 2019 | NDSS | ndss-symposium.org | Provenance triage; −86% false alarms | Heavy post-hoc vs our light filter |
| Julisch, alarm clustering | 2003 | ACM TISSEC 6(4) | 10.1145/950191.950192 [V?] | Root-cause clustering of alarms | Classic aggregation |
| Diab et al., hardware-aware IDS on IoT devices | 2025 | IEEE ETECOM [V?] | arXiv 2512.02272 | LightGBM 75 KB; <30 ms on Raspberry Pi | Edge tree-ensemble comparator |
| Kitsune (see E) | 2018 | NDSS | — | Runs on a Raspberry Pi | Lightweight reference |

---

## What the literature says about OUR novelty (honest)
| Our claim | Already done by others? | How to word it safely |
|---|---|---|
| Single features or metadata can inflate NIDS accuracy | Yes: D'hooge 2022 (dst_port), Jacobs 2022 (TTL) | "Unlike metadata (ports, IPs), the CICIoMT2024 artifacts look like legitimate flow statistics" |
| Using XAI to detect shortcuts | Yes: Lapuschkin 2019; Zhao 2026 (traffic classification) | "We apply attribution-based shortcut detection to IoMT IDS and **quantify** it (31%), triangulated with ablation + recording fingerprinting" |
| Timestamps act as shortcuts in IoT IDS | Concurrent: Roy & Minu 2026 (other dataset, preprint) | "Independently, on CICIoMT2024, and IAT does not look like a timestamp by name" |
| CICIoMT2024 evaluation critique | Limeira 2026 (per-class / kill-chain), who found no artifacts | "First artifact audit of CICIoMT2024" — to be wording-verified after reading Limeira |
| DoS vs DDoS identifiability analysis | Not found | Novel (as far as we found) |
| Mixed-stream protocol for context methods | Not found (Corsini/HELAD hint at the bias) | Novel (as far as we found) |
| m-of-n alert smoothing | Yes: N-BaIoT majority vote | Do not claim as novel; claim the evaluation under interleaving and base rate |
| Lightweight XGBoost for edge | Common | Supporting contribution only |

## Must-read list (in order, ~12 papers)
1. Limeira 2026 (SBSeg). 2. Zhao 2026 ShortcutCatcher. 3. D'hooge 2022 DIMVA. 4. Jacobs 2022 CCS. 5. Arp 2022. 6. Roy & Minu 2026. 7. Flood 2024. 8. Fatima 2026 SLR. 9. Kostas 2025 IoTGeM. 10. Corsini 2021. 11. Neto 2023 CICIoT2023. 12. Axelsson 2000 / Alahmadi 2022.
