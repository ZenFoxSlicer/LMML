"""Point par point - find the word hidden in the Paris listings.

Idea
----
Every planted dot ("letter" dot) was placed on a letter shape in Paris
(latitude/longitude) and given a "colour" taken from what a traveller weighs
before booking: price, reviews per month (popularity) and availability.
For a planted dot, location and colour are both smooth (affine) functions of
its position on the letter, so the planted dots lie on thin 2-D sheets in the
5-D space (log price, log reviews/month, availability, latitude, longitude).
Real listings fill that 5-D space diffusely.

Pipeline
1. Build the 5-D "pointillist" space and standardise it.
2. Separate the letters from the city: distance to the 10th nearest neighbour
   is clearly bimodal; the low mode (dense sheets) are the planted dots.
3. Cluster the planted dots (DBSCAN): each connected sheet is one letter.
4. Look at each letter on its own 2-D principal plane (the sheet itself).
5. Read the letters in the artist's order: by price (cheapest -> dearest).

Outputs: labels.csv, figures/letters.png, figures/separation.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = next(p for p in [ROOT / "data" / "paris_listings.csv", ROOT / "paris_listings.csv"] if p.exists())
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

K = 10                 # neighbour used for the density score
LOG_D_THRESHOLD = -1.4 # valley between the two modes of log(d_10)
DBSCAN_EPS = 0.3
MIN_LETTER_SIZE = 50   # drop tiny fragments

# Letter read from each cluster (by eye, from figures/letters.png) and the
# orientation (swap axes, flip x, flip y) used to draw it upright.
READING = {}  # filled after clustering, see LETTER_BY_PRICE_RANK

d = pd.read_csv(DATA)

# 1. pointillist space -------------------------------------------------------
space = pd.DataFrame({
    "log_price": np.log(d.price),
    "log_reviews_per_month": np.log(d.reviews_per_month),
    "availability_365": d.availability_365,
    "latitude": d.latitude,
    "longitude_km_scaled": d.longitude * np.cos(np.radians(48.86)),
})
X = StandardScaler().fit_transform(space)

# 2. separate letters from the city -----------------------------------------
dist, _ = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X)
log_d = np.log(dist[:, K])
planted = log_d < LOG_D_THRESHOLD

# 3. one cluster per letter --------------------------------------------------
lab = np.full(len(d), -1)
lab[planted] = DBSCAN(eps=DBSCAN_EPS, min_samples=4).fit_predict(X[planted])
sizes = pd.Series(lab[lab >= 0]).value_counts()
lab[np.isin(lab, sizes[sizes < MIN_LETTER_SIZE].index)] = -1

# 5. artist's order: median price of each letter -----------------------------
order = (d.assign(c=lab)[lab >= 0].groupby("c").price.median().sort_values())
rank = {c: i for i, c in enumerate(order.index)}

# letters as read from the figure, in price order (cheapest first)
LETTER_BY_PRICE_RANK = ["N", "E", "U", "R", "O", "N"]
# drawing orientation per price rank: (swap, flip_x, flip_y) of PCA plane
# (PCA axes are arbitrary; these only make the letters easier to read.
#  #1 is the noisiest letter: its colour sits in the densest part of the city,
#  and its map view in the second row shows the N most clearly.)
ORIENT = {0: (False, True, False), 1: (True, False, False),
          2: (False, False, True), 3: (False, True, True),
          4: (False, False, False), 5: (False, False, False)}

d["is_letter"] = lab >= 0
d["letter_cluster"] = [rank.get(c, -1) for c in lab]  # 0..5 in artist's order
d["letter"] = [LETTER_BY_PRICE_RANK[rank[c]] if c >= 0 else "" for c in lab]
d["log_knn10_distance"] = log_d.round(4)
d[["listing_id", "is_letter", "letter_cluster", "letter",
   "log_knn10_distance"]].to_csv(ROOT / "labels.csv", index=False)

word = "".join(LETTER_BY_PRICE_RANK)
print("planted dots:", int(planted.sum()), "| letters kept:", int((lab >= 0).sum()))
print(order.rename("median_price").to_frame().assign(
    letter=LETTER_BY_PRICE_RANK, n=sizes.reindex(order.index).values))
print("WORD:", word, "-> FLAG: SIGMOID_" + word)

# figures ---------------------------------------------------------------------
palette = plt.cm.tab10(np.arange(6))

fig, ax = plt.subplots(1, 3, figsize=(24, 7.5))
ax[0].hist(log_d, bins=120, color="0.4")
ax[0].axvline(LOG_D_THRESHOLD, color="crimson", ls="--")
ax[0].set_title("log distance to 10th neighbour (5-D space)\nlow mode = planted dots")
ax[1].scatter(d.longitude, d.latitude, s=0.4, c="0.75", lw=0)
ax[1].set_title("Paris as given: the letters are invisible")
ax[2].scatter(d.longitude[~d.is_letter], d.latitude[~d.is_letter], s=0.4, c="0.85", lw=0)
for r in range(6):
    m = d.letter_cluster == r
    ax[2].scatter(d.longitude[m], d.latitude[m], s=4, color=palette[r],
                  label=f"{r + 1}: {LETTER_BY_PRICE_RANK[r]}")
ax[2].legend(title="artist's order", markerscale=4)
ax[2].set_title("Separated letter dots (they overlap on the map)")
for a in ax[1:]:
    a.set_aspect(1 / np.cos(np.radians(48.86)))
    a.set_xlabel("longitude"); a.set_ylabel("latitude")
plt.tight_layout()
plt.savefig(FIG / "separation.png", dpi=110)
plt.close()

fig, ax = plt.subplots(2, 6, figsize=(30, 10.5))
for r in range(6):
    m = (d.letter_cluster == r).values
    G = X[m]
    Z = PCA(2).fit_transform(G - G.mean(0))
    Z /= Z.std(0)
    swap, fx, fy = ORIENT[r]
    if swap:
        Z = Z[:, ::-1]
    Z = Z * [(-1 if fx else 1), (-1 if fy else 1)]
    ax[0, r].scatter(Z[:, 0], Z[:, 1], s=10, color=palette[r])
    ax[0, r].set_aspect(1); ax[0, r].set_xticks([]); ax[0, r].set_yticks([])
    p = d.price[m].median()
    ax[0, r].set_title(f"#{r + 1}  '{LETTER_BY_PRICE_RANK[r]}'  (median price {p:.0f} EUR, n={m.sum()})",
                       fontsize=13)
    ax[1, r].scatter(d.longitude[m], d.latitude[m], s=6, color=palette[r])
    ax[1, r].set_aspect(1 / np.cos(np.radians(48.86)))
    ax[1, r].set_title("same dots on the map", fontsize=11)
    ax[1, r].tick_params(labelsize=7)
fig.suptitle(f"Each letter on its own principal plane, in price order  ->  {word}", fontsize=18)
plt.tight_layout()
plt.savefig(FIG / "letters.png", dpi=110)
plt.close()
