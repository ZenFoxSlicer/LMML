"""Fry-ndship: model potato-field edge length and plot the residuals."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
path = sys.argv[1] if len(sys.argv) > 1 else HERE / "potato_fields.csv"
df = pd.read_csv(path)

# Geometry: a rectangle with long side L and area A has short side A/L,
# so its perimeter is 2*(L + A/L).
L = df["long_side_m"]
short = df["area_ha"] * 10_000 / L
df["short_side_m"] = short

# Model: edge = b0 + b1*long_side + b2*short_side  (ordinary least squares)
X = np.column_stack([np.ones(len(df)), L, short])
y = df["edge_m"].to_numpy()
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
df["fitted"] = X @ beta
df["resid"] = y - df["fitted"]

r2 = 1 - (df["resid"] ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"edge = {beta[0]:.2f} + {beta[1]:.3f}*long_side + {beta[2]:.3f}*short_side")
print(f"R^2 = {r2:.4f}")

# Residuals against several candidate axes: the hidden word shows up in one.
axes_to_try = ["fitted", "long_side_m", "short_side_m", "area_ha", "lon", "lat"]
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axs.flat, axes_to_try):
    ax.scatter(df[col], df["resid"], s=3, alpha=0.6)
    ax.axhline(0, color="red", lw=0.8)
    ax.set_xlabel(col)
    ax.set_ylabel("residual (m)")
fig.suptitle(f"Residuals of edge-length model (R² = {r2:.3f})")
fig.tight_layout()
fig.savefig(HERE / "residuals.png", dpi=130)
print("saved residuals.png")

# The word is drawn in residuals vs fitted values: plot it large on its own.
fig, ax = plt.subplots(figsize=(18, 5))
ax.scatter(df["fitted"], df["resid"], s=4, alpha=0.7)
ax.axhline(0, color="red", lw=0.8)
ax.set_xlabel("fitted edge length (m)")
ax.set_ylabel("residual (m)")
ax.set_title(f"Residuals vs fitted (R² = {r2:.3f})")
fig.tight_layout()
fig.savefig(HERE / "residuals_vs_fitted.png", dpi=130)
print("saved residuals_vs_fitted.png")
