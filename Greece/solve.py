"""
Greece - "Just a siblings' thing"

Three oracle models (Ares, Athena, Apollo) predict whether an encounter is a
catastrophic threat (1) or not (0). They all have the SAME accuracy, so accuracy
can't tell them apart. The business rule: missing a real threat (false negative)
is far worse than a false alarm (false positive). So the metric that matters is
RECALL = "of all real threats, how many did the model catch?".

Run (from this folder):  python solve.py
(models/ is unzipped from models-*.zip automatically if missing)
"""
import glob, os, zipfile
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model import MythicThreatClassifier, FEATURE_COLUMNS

THRESHOLD = 0.50
MODELS = ["ares", "athena", "apollo"]

# 0) The weights come zipped; unpack them once.
if not os.path.isdir("models"):
    zipfile.ZipFile(glob.glob("models-*.zip")[0]).extractall(".")

# 1) Load data. Only the 9 numeric columns listed in model.py go into the model,
#    in exactly that order (region / moon_phase are not used by the model).
X_df = pd.read_csv("test_dataset.csv")
y_df = pd.read_csv("test_labels.csv")
data = X_df.merge(y_df, on="encounter_id")          # align rows by id, just in case
X = torch.tensor(data[FEATURE_COLUMNS].values, dtype=torch.float32)
y = data["is_catastrophic_threat"].values
print(f"{len(y)} encounters, {y.sum()} real catastrophic threats\n")

rows = []
for name in MODELS:
    # 2) Build the shared architecture (one linear layer = logistic regression) and load weights.
    model = MythicThreatClassifier()
    model.load_state_dict(torch.load(f"models/model_{name}.pth", map_location="cpu"))
    model.eval()

    # Peek inside: which features does this model actually look at?
    w = model.linear.weight.detach().numpy().ravel()
    used = {f: round(float(v), 4) for f, v in zip(FEATURE_COLUMNS, w) if v != 0}
    print(f"{name.upper():7s} weights used: {used}  bias: {model.linear.bias.item():.4f}")

    # 3) The model outputs a raw score (logit). Sigmoid turns it into a probability 0..1,
    #    and we call it a threat when probability >= 0.50.
    with torch.no_grad():
        prob = torch.sigmoid(model(X)).numpy()
    pred = (prob >= THRESHOLD).astype(int)

    # 4) Metrics. Confusion matrix: TN = calm & said calm, FP = false alarm,
    #    FN = MISSED THREAT (the expensive one), TP = caught threat.
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rows.append({
        "model": name.upper(),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    })

results = pd.DataFrame(rows).set_index("model")
print()
print(results.round(4).to_string(), "\n")

# 5) Decision: highest recall (fewest missed threats). Ties broken by fewer false alarms.
best = results.sort_values(["recall", "FP"], ascending=[False, True]).index[0]
print("Missing a threat costs far more than a false alarm -> pick the highest recall (fewest FN).")
print(f"Selected oracle: {best}")
print(f"FLAG: SIGMOID_{best}")
