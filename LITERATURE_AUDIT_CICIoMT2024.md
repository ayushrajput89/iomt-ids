# Literature audit — CICIoMT2024 IDS papers (draft, 9 Oct 2026)

Collected by web search plus reading the paper pages. **Every row must be verified by reading the paper before it is cited.**

- Entries #7, #12, #13, #17 and #19 come from the abstract only.
- "nr" = not reported in the text that was read.
- Notes in [brackets] are observations, not claims made by the authors.
- Not covered: the full citation list of the dataset paper (Semantic Scholar was blocked), and two papers whose pages were blocked (ICAIIT 2025 Halle; MDPI Information 16:669, UMONS).

| # | Paper | Model | Task / subset | Best acc (F1) | IAT / Number / Weight | Split | XAI / top features |
|---|---|---|---|---|---|---|---|
| 1 | Mohammadi et al. 2024, ICIS (IEEE), arXiv 2410.23306 | 1D-CNN | 2 / 6 / 19 | 0.99 for all three tasks; F1 type not stated | nr | nr | none |
| 2 | Gueriani et al., arXiv 2604.06254 (SE-ViT-BiLSTM) | ViT + BiLSTM | 6-class, subset of an augmented version | 98.16% | 83 selected features (method not explained); drops frame.time | 80/20 | none |
| 3 | Gueriani et al., arXiv 2508.12470 (BiGAT-ID) | BiGRU-LSTM-attention | 6-class | 99.13% (macro-F1 ~99) | Features look CICFlowMeter-style (83) | random 80/20 | SHAP, top feature = **Timestamp** [time shortcut] |
| 4 | Jaiswal et al., arXiv 2604.03205 | Tsetlin Machine vs RF/XGB etc. | BT binary; WiFi+MQTT 6-class | TM 0.995 BT; RF/XGB 0.998; 6-class 0.907 | IAT, Number in feature list | official files + 5-fold CV (unclear) | TM clause maps |
| 5 | Shaikh et al. 2025, Frontiers in Medicine 12 | CNN-LSTM + DQN/PPO | binary + multiclass (19 rows) | binary 99.58 (mF1 0.986); **multiclass 77.73 (mF1 0.583)** | MIFS 42→34; artifacts not named | stratified random 70/10/20 [feature selection done "independently on train and test": possible leakage] | none |
| 6 | **Limeira et al. 2026, SBSeg** ("Beyond Aggregate Accuracy") | RF, LGBM, 1D-CNN, AE | 2 / 6 / 19, WiFi+MQTT | 19-class acc 0.9967, **macro-F1 0.9207** | **All 45 kept, IAT/Number/Weight listed** | **Official split** | SHAP + LIME; IAT among top RF features. Checked for artifacts and concluded none were "obvious" |
| 7 | Mohammad & Abdulrahman 2026, Ibn Al-Haitham J. | DNN, GRU-DNN | 2 / 6 / 19 | 98.4% (19-class) | nr | nr | nr |
| 8 | Misbah et al. 2025, IJACSA 16(2) | FL + RF/AdaBoost/SVM, stacking | nr | 99.22% | 45 features; use of the three unconfirmed | nr | none |
| 9 | AL-Mashakbah et al. 2026, IJACSA 17(8) | FL MLP | 4/5-class | 99.21 (mF1 98.74) | includes "Inter-Arrival Time Mean" | nr | none |
| 10 | (Oujda/UPJV), arXiv 2509.10563 | RF, CatBoost | nr | 99.92% | IAT used | random 70/30 | SHAP + LIME; **RF top feature = IAT**, then rst_count |
| 11 | Berrezzek et al., arXiv 2608.00869 | RF, LGBM | binary | 0.996 | no manual feature removal | random 70/30 | SHAP + LIME; **LIME top feature = IAT** |
| 12 | Winanto et al. 2026, JUTIF 7(3) | Transformer | 19-class | 98.76 (mF1 0.92) | nr | stratified | nr |
| 13 | Akar et al. 2025, IEEE Access 13 (L2D2) | LSTM | 19-class | ~98% | nr | nr | nr |
| 14 | D'Ambrosio et al. 2026, IFIP Networking | CNN, LSTM | 6-class | 96.13 | Per-packet header features, not the CSV features | nr | Gradient SHAP |
| 15 | Alharbi et al. 2025, AIMS Mathematics 10(11) | voting ensemble | 2 / 6 / 19 | 19-class 98.46 (mF1 89.20) | **All 44 kept; IAT is the top RF importance by a wide margin** | random 80/20 | Gini importance |
| 16 | Mortatha & Abdulah 2025, IJSSE 15(10) | WOA + XGB/LGBM | binary + 14-class | 99.76 (mF1 0.96) | WOA reduces to 18 features | own 80/20 PCAP-level split | none |
| 17 | Rehman et al. 2025, ICISSP | XGB, KNN, DT, RF | binary + multi | nr | filter feature selection | nr | nr |
| 18 | Arifin & Martadinata 2025, IAENG IJCS 52(4) | RF etc. | 19-class | 99% | nr | 85/15 | none |
| 19 | Riyadi et al. 2025, JIOS 49(2) | XGB + SMOTEENN | nr | 99.81 | IG + PCA | nr | nr |
| 20 | Alami-Kamouri et al. 2026, IJACSA 17(7) | AE + RF | 19-class | RF 98.95; AE-RF 94.60 (mF1) | 44 features → latent | 80/20 + 5-fold | SHAP on latent dimensions |

## Counts (20 papers)

| Item | Count | Papers |
|---|---|---|
| ≥99% accuracy reported | 13 | |
| Explicitly all features incl. IAT/Number/Weight | 2 | #6, #15 |
| IAT shown as the #1 feature | 3 | #10, #11, #15 |
| Macro-F1 reported | 8 | |
| Official train/test split, clearly | 1 | #6 |
| SHAP/LIME used | 6 | |

## Leakage / shortcut related work

- **Limeira 2026 (#6)** is the closest existing work. They checked for artifacts and concluded there were none, even though IAT is among their top SHAP features. Their 19-class macro-F1 of 0.92 with all features matches our all-feature 89–91. We must cite and differentiate.
- **Tolay, arXiv 2608.10349 (Internet of Things), CICIoT2023:** deduplication; Number has KS 0.939 and IAT KS 0.793; removing Number left results "virtually unchanged". They do not call it leakage.
- **Ozechi & Okonkwoabutu, arXiv 2603.22771, CICIoT2023:** IAT is the top SHAP feature; no shortcut analysis.
- **Roy & Minu, arXiv 2609.28725, CyberFlowIoT-GICAP:** shows timestamp shortcut learning (LGBM macro-F1 99.28 → 92.58). Methodological related work.
- No paper found that explicitly calls IAT in CICIoMT2024/CICIoT2023 a timestamp-like leakage feature. The gap looks open, but the search was incomplete.
