# BELGIUM: find a hidden word in the errors of a prediction.
# 1. Read the field measurements and calculate each field's short side.
# 2. Fit a formula that predicts edge length from the declared measurements.
# 3. Subtract each prediction from its real measurement.
# 4. Plot these differences: their arrangement reveals the word.
# A residual means "actual value minus predicted value".

"""Fry-ndship: model potato-field edge length and find Mariette's word.

The co-op's analyst predicted edge length from what farmers declare:
the field's dimensions (long side, short side = area / long side) and the
ditch length. That model explains >99.99% of the variation, yet its
residuals are not noise: plotted against the fitted values they spell a
word. The structure left over is exactly what the declarations miss,
the hedge Mariette noted on site (hedge_m ~ 0.08 * (long + short) plus the
same lettering), so the word is Mariette's observation: SUPERVISION.

Other layers are decoys: long + short alone gives REGRESSION (carried by
the farmer-declared ditch_m), and adding Mariette's hedge_m leaves
PREDICTION in the last 0.1 m of rounding-level residuals.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use the CSV named on the command line, or the CSV beside this script.
HERE = Path(__file__).resolve().parent
path = sys.argv[1] if len(sys.argv) > 1 else HERE / "potato_fields.csv"
df = pd.read_csv(path)

# Geometry: a rectangle with long side L and area A has short side A/L.
# One hectare is 10,000 square meters.
# For a rectangular field: short side = area / long side.
df["short_side_m"] = df["area_ha"] * 10_000 / df["long_side_m"]


# Fit one straight-line formula with several inputs.
# target is the column to predict; cols lists the columns we use to predict it.
def ols(target, cols):
    """Least-squares fit of target on cols (+ intercept)."""
    # Make one row per field and one column per input.
    # The extra column of ones lets the formula learn a constant starting value.
    X = np.column_stack([np.ones(len(df))] + [df[c] for c in cols])
    # These are the real measurements that the formula should match.
    y = df[target].to_numpy()
    # Choose the coefficients that make the total squared prediction error smallest.
    # beta contains those coefficients; *_ ignores the other returned information.
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    # @ multiplies the input table by the coefficients to calculate all predictions.
    fitted = X @ beta
    # Positive residual: the real value is larger than the prediction.
    # Negative residual: the real value is smaller.
    resid = y - fitted
    # R-squared compares our errors with simply guessing the average every time.
    # A value close to 1 means a close fit, but errors can still contain a pattern.
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"{target} ~ {cols}: coef {dict(zip(['intercept'] + cols, beta.round(3)))}, R^2 = {r2:.6f}")
    return fitted, resid, r2


# Analyst's model: edge from the farmer-declared dimensions and ditch.
fitted, resid, r2 = ols("edge_m", ["long_side_m", "short_side_m", "ditch_m"])
# Mariette's hedge observation carries the same pattern.
h_fitted, h_resid, _ = ols("hedge_m", ["long_side_m", "short_side_m"])

# Make two plots: edge-length errors above, hedge-length errors below.
fig, axs = plt.subplots(2, 1, figsize=(18, 9))
# Each dot represents one field. Its x-position is the prediction;
# its y-position is the part that the formula did not explain.
axs[0].scatter(fitted, resid, s=4, alpha=0.7)
axs[0].set_title(f"edge ~ long + short + ditch (R² = {r2:.5f}): residuals vs fitted")
axs[0].set_xlabel("fitted edge length (m)")
axs[1].scatter(h_fitted, h_resid, s=4, alpha=0.7, color="tab:green")
axs[1].set_title("hedge_m (noted by Mariette) ~ long + short: residuals vs fitted")
axs[1].set_xlabel("fitted hedge length (m)")
# The red zero line marks a perfect prediction on both plots.
for ax in axs:
    ax.axhline(0, color="red", lw=0.8)
    ax.set_ylabel("residual (m)")
fig.suptitle("Mariette's word: SUPERVISION")
fig.tight_layout()
# Save the plots beside this script so the hidden lettering can be inspected.
fig.savefig(HERE / "residuals_vs_fitted.png", dpi=130)
print("saved residuals_vs_fitted.png")
