from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent

guide = pd.read_csv(DATA_DIR / "office_price_guide.csv")
pur = pd.read_csv(DATA_DIR / "purchases.csv")

df = pur.merge(guide, on="Product", how="left", validate="many_to_one")
df["unit_price"] = (df["Total_paid_eur"] / df["Quantity"]).round(6)

ok = df[
    (df["unit_price"] >= df["Fair_min_eur"] - 1e-6)
    & (df["unit_price"] <= df["Fair_max_eur"] + 1e-6)
].copy()

print("receipts:", len(df), "valid:", len(ok))
print("city == home_city on valid rows:", (ok["City"] == ok["Home_city"]).all())
print("pilots:", sorted(pur["Pilot_no"].unique()))

board = np.zeros((6, 21), dtype=int)
for list_no, pilot_no in zip(ok["List_no"], ok["Pilot_no"]):
    board[int(list_no) - 1, int(pilot_no) - 1] = 1

print(board)
for row in board:
    print("".join("#" if v else "." for v in row))

labels = guide.sort_values("List_no")["Product"].tolist()

fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(board, cmap="binary", aspect="equal", interpolation="nearest")
ax.set_xticks(range(21), [str(p) for p in range(1, 22)])
ax.set_yticks(range(6), [f"{i+1}. {n}" for i, n in enumerate(labels)])
ax.set_xticks(np.arange(-0.5, 21, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
ax.grid(which="minor", color="#cccccc", linewidth=0.6)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("Pilot number")
ax.set_title("Christmas board: fair-priced souvenirs per pilot")
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "board.png", dpi=160)
print("saved board.png")
