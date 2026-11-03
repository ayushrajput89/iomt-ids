# ==== CELL 7 (v10a follow-up, CPU, ~5-10 min) ====
# (a) Per-class F1 at 19 vs 13 granularity - which classes still fail after DoS+DDoS merge?
# (b) Fingerprint "all files" re-check with RandomForest (the XGB all-files result looked broken)
# Run in the same Colab session after Cells 1-6 (uses LAB, LABELS, type_merged, FP, SHORTCUT, FREE39, OUT_DIR).
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ---- (a) per-class F1, static official test ----
z = np.load(os.path.join(OUT_DIR, 'probs_seed42.npz'), allow_pickle=True)
yt, yp = LAB[z['y_static']], LAB[z['P_static'].argmax(1)]
f19 = pd.Series(f1_score(yt, yp, labels=LABELS, average=None), index=LABELS)
tm = np.vectorize(type_merged)
T13 = sorted(set(tm(LAB)))
f13 = pd.Series(f1_score(tm(yt), tm(yp), labels=T13, average=None), index=T13)
PC = pd.DataFrame({'F1_19class': f19, 'F1_13type': f19.index.map(lambda l: f13[type_merged(l)])}).mul(100).round(2)
PC['support'] = pd.Series(yt).value_counts().reindex(LABELS).values
PC = PC.sort_values('F1_13type'); PC.to_csv(os.path.join(OUT_DIR, 'P1_per_class_19_vs_13.csv'))
print('Per-class F1 (%), static official test, sorted by 13-type F1:'); print(PC.to_string())

# ---- (b) fingerprint re-check, all 51 files, RandomForest ----
sub = FP.sample(frac=1.0, random_state=42); sub = sub[sub.groupby('file').cumcount() < 3000]
f2i = {f: i for i, f in enumerate(sorted(sub['file'].unique()))}; y = sub['file'].map(f2i).values
a, b = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=42)
rows = []
for name, feats in [('IAT only', ['IAT']), ('Number + Weight', ['Number', 'Weight']),
                    ('IAT + Number + Weight', SHORTCUT), ('Shortcut-free 39 (reference)', FREE39)]:
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(sub[feats].values[a], y[a])
    rows.append({'features': name, 'file_id_acc_%': round(100 * accuracy_score(y[b], rf.predict(sub[feats].values[b])), 2),
                 'chance_%': round(100 / len(f2i), 2)})
    print('done:', name)
FR = pd.DataFrame(rows); FR.to_csv(os.path.join(OUT_DIR, 'TABLE_F2_fingerprint_allfiles_RF.csv'), index=False)
print(f'\nWhich of the {len(f2i)} PCAP files did this row come from? (RandomForest, 3000 rows/file)'); print(FR.to_string(index=False))
