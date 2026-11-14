"""Fry-ndship: model potato-field edge length and plot the residuals.

The residuals hide a word at every level of the model. Each extra column
explains the previous word away and reveals the next one:
  long + short sides                    -> REGRESSION   (decoy)
  + ditch_m (farmer's declaration)      -> SUPERVISION  (decoy)
  + hedge_m (Mariette's own observation) -> PREDICTION
With all four terms the fit is exact
(edge = 0.82*(long + short) + ditch + hedge, intercept 0), and the
remaining +-0.1 m residuals spell the word Mariette observed: PREDICTION.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
path = sys.argv[1] if len(sys.argv) > 1 else HERE / "potato_fields.csv"
df = pd.read_csv(path)

# Geometry: a rectangle with long side L and area A has short side A/L.
df["short_side_m"] = df["area_ha"] * 10_000 / df["long_side_m"]

# Model: edge = b0 + b1*long + b2*short + b3*ditch + b4*hedge  (least squares)
cols = ["long_side_m", "short_side_m", "ditch_m", "hedge_m"]
X = np.column_stack([np.ones(len(df))] + [df[c] for c in cols])
y = df["edge_m"].to_numpy()
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
df["fitted"] = X @ beta
df["resid"] = y - df["fitted"]

r2 = 1 - (df["resid"] ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("coefficients:", dict(zip(["intercept"] + cols, beta.round(4))))
print(f"R^2 = {r2:.8f}")

# Residuals vs fitted values: the points spell the hidden word.
fig, ax = plt.subplots(figsize=(18, 5))
ax.scatter(df["fitted"], df["resid"], s=4, alpha=0.7)
ax.axhline(0, color="red", lw=0.8)
ax.set_xlabel("fitted edge length (m)")
ax.set_ylabel("residual (m)")
ax.set_title("Residuals vs fitted: edge ~ long + short + ditch + hedge  ->  PREDICTION")
fig.tight_layout()
fig.savefig(HERE / "residuals_vs_fitted.png", dpi=130)
print("saved residuals_vs_fitted.png")
