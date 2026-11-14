# SPAIN: turn shopping receipts into a picture.
# 1. Read the receipts and the list of fair prices.
# 2. Keep receipts whose price per item is fair.
# 3. Mark their positions on a 6-row, 21-column board.
# 4. Save the board as board.png so its pattern can be read.
# This file runs from top to bottom.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Draw directly to a file; do not open a chart window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Find the folder containing this script, even if we run it from elsewhere.
BASE_DIR = Path(__file__).resolve().parent

# pandas reads each CSV as a table with named columns.
# guide contains allowed prices; pur contains the actual purchases.
guide = pd.read_csv(BASE_DIR / "office_price_guide.csv")
pur = pd.read_csv(BASE_DIR / "purchases.csv")

# Attach the matching price-guide row to every purchase using Product.
# Many receipts may use one guide row, but duplicate guide entries are an error.
df = pur.merge(guide, on="Product", how="left", validate="many_to_one")
# Divide the receipt total by the number of items to get the price of one item.
df["unit_price"] = (df["Total_paid_eur"] / df["Quantity"]).round(6)

# Keep rows inside the allowed price range. & means both checks must pass.
# The tiny 1e-6 allowance avoids rejecting a price because of rounding.
ok = df[
    (df["unit_price"] >= df["Fair_min_eur"] - 1e-6)
    & (df["unit_price"] <= df["Fair_max_eur"] + 1e-6)
].copy()

# Print quick checks so we can see how many receipts survived the filter.
# The city comparison is printed for inspection; it does not filter any rows.
print("receipts:", len(df), "valid:", len(ok))
print("city == home_city on valid rows:", (ok["City"] == ok["Home_city"]).all())
print("pilots:", sorted(pur["Pilot_no"].unique()))

# Start with an empty board: 0 means empty, and 1 will mean filled.
board = np.zeros((6, 21), dtype=int)
for list_no, pilot_no in zip(ok["List_no"], ok["Pilot_no"]):
    # List number chooses the row; pilot number chooses the column.
    # Subtract 1 because Python starts counting positions at 0.
    board[int(list_no) - 1, int(pilot_no) - 1] = 1

# Show the same board as numbers and as a simple # / . picture.
print(board)
for row in board:
    print("".join("#" if v else "." for v in row))

# Put the product names in list-number order for the row labels.
labels = guide.sort_values("List_no")["Product"].tolist()

# Draw the board. Each filled cell becomes a black square.
fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(board, cmap="binary", aspect="equal", interpolation="nearest")
ax.set_xticks(range(21), [str(p) for p in range(1, 22)])
ax.set_yticks(range(6), [f"{i+1}. {n}" for i, n in enumerate(labels)])
# Place the thin grid lines between cells, not through their centers.
ax.set_xticks(np.arange(-0.5, 21, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
ax.grid(which="minor", color="#cccccc", linewidth=0.6)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("Pilot number")
ax.set_title("Christmas board: fair-priced souvenirs per pilot")
# Make room for the labels, then save the finished picture beside the script.
fig.tight_layout()
fig.savefig(BASE_DIR / "board.png", dpi=160)
print("saved board.png")
