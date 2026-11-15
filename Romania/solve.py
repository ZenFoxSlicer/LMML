"""Romania - "Shaken, Still Standing".

Decide, for each of the 5,000 sealed seismic records, whether it is a deep Vrancea
earthquake (1) or something else, like a truck, a quarry blast or construction (0).

What this script does, step by step:
  1. Load the labelled training records and the unlabelled ("sealed") ones.
  2. Read from the evaluator's vault how accurate the predictions must be.
  3. Baseline: try ordinary models one at a time on all records, plus a plain
     ensemble. This is the evidence that no single model is good enough.
  4. Find the hidden networks. Records come from sensor networks with different
     amplification (gain). The script measures each record's gain, splits the records
     into groups, and removes the gain, so every record is on the same scale.
  5. For each group, try several models and ensembles (stacking), keep the best one
     for that group ("mixture of experts": one specialist per network).
  6. Predict the sealed records, write predictions.csv, run the official evaluator.

    python solve.py

Only needs numpy, pandas, scipy and scikit-learn.
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")          # keep the output readable (library version notices)

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
CHANNELS = [f"c{j + 1:02d}" for j in range(48)]   # the 48 spectral-amplitude columns
SEED = 0                                           # fixed randomness, so every run gives the same result


# ============================================================================= 1. load the data
# train.csv  : 7,500 records with the answer known (column "vrancea": 1 = earthquake, 0 = not)
# sealed.csv : 5,000 records with no answer; these are the ones we have to predict
train = pd.read_csv(DATA / "train.csv")
sealed = pd.read_csv(DATA / "sealed.csv")
y = train["vrancea"].to_numpy()                    # the answer for each training record
ALL = pd.concat([train, sealed])[CHANNELS].to_numpy()   # all records' 48 numbers, training first
n_train = len(train)
print(f"training records: {n_train}   sealed records: {len(sealed)}   "
      f"share of earthquakes in training: {y.mean():.1%}")


# ============================================================================= 2. how good is "good enough"?
# The evaluator hides the flag behind an error-correcting code (Reed-Solomon). It does not
# need every prediction to be right: it recovers the flag when at least (n + k) / 2 of the
# n predictions are correct. n and k are stored in the vault, so we can read the bar.
# The vault does not reveal the labels.
vault = np.load(DATA / "vault.npz")
n, k = int(vault["n"]), int(vault["k"])
NEEDED = (n + k) / 2 / n
print(f"the evaluator needs about {NEEDED:.1%} of the sealed records right "
      f"({int(np.ceil((n + k) / 2))} of {n})\n")

# Cross-validation is a fair exam. Split the labelled records into 5 parts, train on 4,
# test on the 5th, rotate through all 5. Every record gets a prediction from a model that
# never saw it, which is what will happen with the sealed records.
def cv(seed=SEED):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def oof_correct(model, X, labels):
    """For each record: did the model predict it correctly while the record was hidden?"""
    return cross_val_predict(model, X, labels, cv=cv(), n_jobs=-1) == labels


# ============================================================================= 3. baseline: single models on everything
# Each model is a different "expert" with a different idea of what an earthquake looks like:
#   logistic regression  one straight boundary: weighted sum of the channels vs a threshold.
#   SVM (RBF kernel)     a smooth curved boundary, based on similarity to known examples.
#   random forest        many decision trees ("if c35 < 0.2 and c43 > 1 then ...") that vote.
#   gradient boosting    trees built one after another, each fixing the previous ones'
#                        mistakes. Strong at threshold rules and channel interactions.
#   neural network       layers of small adjustable units that can learn many shapes.
# StandardScaler puts each channel on the same scale (mean 0, spread 1). Models that use
# distances or weights need that. Trees don't.
def lr():  return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
def svm(): return make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale"))
def rf():  return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)
def mlp(): return make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True, random_state=SEED))
def hgb(): return HistGradientBoostingClassifier(
    max_iter=1500, learning_rate=0.03, early_stopping=True, n_iter_no_change=50, random_state=SEED)


X_raw = ALL[:n_train]
print("STEP 3 - baseline, one model at a time on all training records (5-fold accuracy):")
for name, make in [("logistic regression", lr), ("SVM (rbf)", svm), ("random forest", rf),
                   ("gradient boosting", hgb), ("neural network", mlp)]:
    print(f"  {name:22s} {oof_correct(make(), X_raw, y).mean():.3f}", flush=True)
print("  -> every single model is well below the bar.\n")


# ============================================================================= 4. find the hidden networks
# What the data shows: 36 of the 48 channels rise and fall TOGETHER within a record. They
# are scaled by a shared per-record factor, the station's amplification ("gain"). The other
# 12 channels don't behave that way. These gain channels have heavier tails than a normal
# bell curve (positive kurtosis), which is how we find them automatically.
gain_ch = kurtosis(ALL, axis=0) > 0.5
center = np.median(ALL, axis=0)
spread = np.median(np.abs(ALL - center), axis=0)

# A record's gain = how far, on average, its gain channels sit from the typical value.
gain = np.sqrt(np.mean(((ALL[:, gain_ch] - center[gain_ch]) / spread[gain_ch]) ** 2, axis=1))

# The gain values form clearly separated clumps, one per network type. K-means finds the
# clumps on a log scale, because gain is a multiplication factor.
km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(np.log(gain)[:, None])
order = np.argsort(km.cluster_centers_.ravel())             # group 0 = low gain, group 1 = high gain
group = np.argsort(order)[km.labels_]
g_train, g_sealed = group[:n_train], group[n_train:]

# Remove the gain: divide the gain channels by the record's gain. After this, a quiet
# network's record and a loud network's record are on the same scale.
NORM = ALL.copy()
NORM[:, gain_ch] = (ALL[:, gain_ch] - center[gain_ch]) / gain[:, None]

# One table holding every view of each record: columns 0-47 raw, columns 48-95 gain-free,
# column 96 the (log) gain itself, so a model can still tell which network a record came from.
BOTH = np.hstack([ALL, NORM, np.log(gain)[:, None]])
RAW_COLS, NORM_COLS, GAIN_COL = list(range(48)), list(range(48, 96)), [96]
print("STEP 4 - hidden networks found from the gain:")
for gi in (0, 1):
    print(f"  group {gi} ({'low' if gi == 0 else 'high'} gain, about x{np.median(gain[group == gi]):.2f}): "
          f"{(g_train == gi).sum()} training, {(g_sealed == gi).sum()} sealed records")
print("  (note: the sealed file has a different mix of networks than the training file)\n")


# ============================================================================= 5. specialist per network
# A model that only looks at some columns: "raw" or "gain-free".
def on(cols, model):
    return make_pipeline(ColumnTransformer([("pick", "passthrough", cols)]), model)


def stack(final):
    # Stacking: a "manager" model on top of several experts. Each expert predicts every
    # training record while that record is hidden from it, so the manager learns from honest
    # opinions. The manager learns how far to trust each expert. With passthrough=True it also
    # sees the channels, so the trust can depend on the kind of record.
    return StackingClassifier(
        estimators=[("lr", on(NORM_COLS, lr())), ("svm_norm", on(NORM_COLS, svm())),
                    ("svm_raw", on(RAW_COLS, svm())), ("hgb", on(RAW_COLS, hgb())),
                    ("rf", on(RAW_COLS, rf()))],
        final_estimator=final, cv=cv(SEED + 1), passthrough=final is not None and
        not isinstance(final, LogisticRegression), n_jobs=-1)


CANDIDATES = {
    "SVM on gain-free channels": lambda: on(NORM_COLS, svm()),
    "gradient boosting on raw channels": lambda: on(RAW_COLS, hgb()),
    "stacking, linear manager": lambda: stack(LogisticRegression(max_iter=5000)),
    "stacking, boosting manager": lambda: stack(hgb()),
}

# Tried and dropped in an earlier run: "pooled" models trained on all 7,500 gain-free records
# at once. They did worse (quiet network 0.757, loud 0.870), which shows the networks follow
# DIFFERENT rules, not just different volumes. So each network keeps its own specialist.

X_both, X_sealed_both = BOTH[:n_train], BOTH[n_train:]


def best_specialist(mask, label):
    """Exam every candidate on the training records in `mask`; return (best name, its score)."""
    scores = {}
    for name, make in CANDIDATES.items():
        scores[name] = oof_correct(make(), X_both[mask], y[mask]).mean()
        print(f"  {label:28s} {name:36s} {scores[name]:.3f}", flush=True)
    best = max(scores, key=scores.get)
    print(f"  {label:28s} -> best: {best} ({scores[best]:.3f})", flush=True)
    return best, scores[best]


# A "segment" is a set of records served by one specialist:
# (name, training mask, sealed mask, chosen model, cross-validated score)
segments = []
print("STEP 5 - one specialist per network, with a learned gate inside each group:")
for gi in (0, 1):
    m_tr, m_se = g_train == gi, g_sealed == gi
    label = f"group {gi}"
    whole = best_specialist(m_tr, label)

    # ---- the learned gate --------------------------------------------------------------
    # The story has three networks, but the gain shows only two groups, so one group may
    # still mix two networks with different rules. Where one hides, a straight-line expert
    # and a tree expert succeed on DIFFERENT records. Collect the records where exactly one
    # of the two is right, and let a one-question decision tree ("is channel X above t?")
    # learn what separates them. That question is the gate.
    lin_ok = oof_correct(on(NORM_COLS, lr()), X_both[m_tr], y[m_tr])
    tree_ok = oof_correct(on(RAW_COLS, hgb()), X_both[m_tr], y[m_tr])
    disagree = lin_ok != tree_ok
    # The gate looks at the GAIN-FREE channels: on a quiet network the raw values are squeezed
    # about 4x, so a boundary like "c13 above t" only lines up across networks after the
    # gain is removed. min_samples_leaf keeps both sides big enough to train a specialist.
    gate_cols = NORM_COLS
    gate = DecisionTreeClassifier(max_depth=1, min_samples_leaf=50, random_state=SEED).fit(
        X_both[m_tr][disagree][:, gate_cols], lin_ok[disagree])
    feat, thr = gate.tree_.feature[0], gate.tree_.threshold[0]
    feat_name = CHANNELS[feat] + " (gain-free)"
    col = gate_cols[feat]
    side_tr, side_se = BOTH[:n_train, col] > thr, BOTH[n_train:, col] > thr
    print(f"  {label}: gate found -> '{feat_name} > {thr:.2f}' "
          f"(linear expert right on {lin_ok.mean():.3f}, tree expert on {tree_ok.mean():.3f})")

    # Score a specialist on each side of the gate. Keep the split only if it beats one
    # specialist for the whole group (weighted by how many records fall on each side).
    halves, split_score = [], 0.0
    for side in (False, True):
        mt, ms = m_tr & (side_tr == side), m_se & (side_se == side)
        hl = f"{label} {feat_name}{'>' if side else '<='}{thr:.2f}"
        if mt.sum() < 150:                       # too few records to train a specialist
            split_score = -1
            break
        name, sc = best_specialist(mt, hl)
        halves.append((hl, mt, ms, name, sc))
        split_score += sc * mt.sum() / m_tr.sum()
    if split_score > whole[1]:
        print(f"  {label}: split kept ({split_score:.3f} vs {whole[1]:.3f} unsplit)\n")
        segments += halves
    else:
        print(f"  {label}: split rejected ({split_score:.3f} vs {whole[1]:.3f} unsplit)\n")
        segments.append((label, m_tr, m_se, whole[0], whole[1]))

# Expected accuracy on the sealed file: weight each segment's score by its share there.
expected = sum(sc * ms.mean() for _, _, ms, _, sc in segments)
print(f"expected sealed accuracy: {expected:.3f} (bar {NEEDED:.3f})\n")


# ============================================================================= 6. predict and evaluate
# Retrain each segment's best approach on ALL that segment's training records, then route
# every sealed record to the specialist of its own segment.
pred = np.zeros(len(sealed), dtype=int)
for label, mt, ms, name, _ in segments:
    if ms.any():
        model = CANDIDATES[name]().fit(X_both[mt], y[mt])
        pred[ms] = model.predict(X_sealed_both[ms])

out = pd.DataFrame({"record_id": sealed["record_id"], "vrancea": pred})
out.to_csv(HERE / "predictions.csv", index=False)
print(f"wrote predictions.csv: {int(pred.sum())} earthquakes, {int(len(pred) - pred.sum())} other")

# The official evaluator prints SIGMOID_<WORD> only when enough predictions are correct.
print("\nevaluator says:")
sys.stdout.flush()
subprocess.run([sys.executable, str(HERE / "evaluation_script.py"), str(HERE / "predictions.csv")],
               check=False)
