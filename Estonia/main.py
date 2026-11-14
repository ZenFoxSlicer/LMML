"""
Estonia e-Residency fraud screen -- audit of the reported 93.42% accuracy.

Three questionable choices in project_pipeline.py:
  1. StandardScaler is fitted on ALL rows before the split      (decoy)
  2. A single 80/20 split with a hand-picked seed (1137)         (decoy)
  3. A random row-level split that ignores applicant identity    (REAL BUG)

The same person (same date_of_birth + citizenship) can apply several times
with nearly identical features. A random split puts one copy in train and
another in test, and 1-NN simply finds the twin and copies its label.

Fix: keep identity fields OUT of the features, but use them as CV groups,
so all of one person's applications are held out together.

Run:  python main.py   (from the Estonia folder)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    LeaveOneGroupOut,
    LeaveOneOut,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- load data
df = pd.read_csv("applications.csv")

IDENTITY = ["submission_id", "date_of_birth", "citizenship"]
X = df.drop(columns=IDENTITY + ["label"]).to_numpy()
y = df["label"].to_numpy()

# One group per real person: same birth date + same citizenship.
groups = df.groupby(["date_of_birth", "citizenship"]).ngroup().to_numpy()


def model():
    # The shipped model: 1-nearest-neighbour on standardised features.
    return KNeighborsClassifier(n_neighbors=1)


def leaky_scaled_X():
    # What the original pipeline does: scaler sees every row, test included.
    return StandardScaler().fit_transform(X)


def accuracy(pred):
    return float((pred == y).mean())


# ------------------------------------------------------ evidence: duplicates
print("=== Data check: repeat applicants ===")
sizes = pd.Series(groups).value_counts()
print(f"rows: {len(df)}   distinct identities: {sizes.size}")
print("applications per identity:", sizes.value_counts().sort_index().to_dict())
print(f"rows belonging to a repeat applicant: {(sizes[sizes > 1].sum())}")

# How often is a row's nearest OTHER row the same person?
Xs = leaky_scaled_X()
nn = NearestNeighbors(n_neighbors=2).fit(Xs)
dist, idx = nn.kneighbors(Xs)
nearest = idx[:, 1]  # column 0 is the row itself
same_person = groups[nearest] == groups
print(f"nearest neighbour is the same person: {same_person.mean():.2%}")
print(f"median NN distance  same person: {np.median(dist[same_person, 1]):.4f}"
      f"   different person: {np.median(dist[~same_person, 1]):.4f}")
print()

# ------------------------------------------------------------- experiments
print("=== Accuracy, one change at a time (1-NN) ===")
results = {}

# Baseline: original choices, but evaluated with LOO as the task requires.
pred = cross_val_predict(model(), leaky_scaled_X(), y, cv=LeaveOneOut())
results["baseline (scaler on all rows, LOO)"] = accuracy(pred)

# Decoy 1 fixed: scaler is refitted inside every fold.
pred = cross_val_predict(
    make_pipeline(StandardScaler(), model()), X, y, cv=LeaveOneOut()
)
results["decoy 1 fixed: scaler inside CV"] = accuracy(pred)

# Real bug fixed ALONE: grouped CV, scaler still leaky (isolates the effect).
pred = cross_val_predict(
    model(), leaky_scaled_X(), y, cv=LeaveOneGroupOut(), groups=groups
)
results["real bug fixed alone: grouped CV"] = accuracy(pred)

for name, acc in results.items():
    print(f"{name:<40} {acc:.4f}")

# Decoy 2: the single hand-picked split. Re-run the ORIGINAL holdout over many
# seeds to show 1137 was not special -- the average sits in the same place.
print()
print("=== Decoy 2: original 80/20 holdout over many seeds ===")
scores = []
for seed in range(200):
    Xtr, Xte, ytr, yte = train_test_split(
        leaky_scaled_X(), y, test_size=0.2, random_state=seed
    )
    scores.append((model().fit(Xtr, ytr).predict(Xte) == yte).mean())
scores = np.array(scores)
Xtr, Xte, ytr, yte = train_test_split(
    leaky_scaled_X(), y, test_size=0.2, random_state=1137
)
reported = (model().fit(Xtr, ytr).predict(Xte) == yte).mean()
print(f"seed 1137 (reported): {reported:.4f}")
print(f"200 seeds: mean {scores.mean():.4f}  std {scores.std():.4f}  "
      f"range [{scores.min():.4f}, {scores.max():.4f}]")
print()

# ------------------------------------------------------- corrected pipeline
# Both legitimate fixes together: scaler inside CV + identity-grouped LOO.
print("=== Corrected pipeline ===")
corrected = make_pipeline(StandardScaler(), model())
pred = cross_val_predict(corrected, X, y, cv=LeaveOneGroupOut(), groups=groups)
true_acc = accuracy(pred)
print(f"true accuracy (grouped LOO): {true_acc:.4f}")
print(f"FLAG: SIGMOID_{int(true_acc * 100)}")
