"""Point par point: find and draw the letters hidden among the Paris listings.

The puzzle in plain words
-------------------------
The spreadsheet has ~28,000 rows. Each row is a place to stay in Paris (a "listing").
An artist secretly added extra rows. If you look at the right columns, those extra
rows line up into shapes of letters, while the real listings are just a messy cloud.

What this script does
---------------------
1. Picks the 5 columns a traveller cares about (location, price, how often it gets
   reviews, how many nights are free) and makes them comparable to each other.
2. Looks for places where rows are packed unusually tightly together. The real
   listings are spread out; the artist's rows are packed into thin lines (the letter
   strokes). Each tight group = one letter.
3. Draws each group as a picture so a human can read the letter, puts the letters
   in order (cheapest to most expensive), and saves which row belongs to which letter.
"""

# ---------------------------------------------------------------- tools we use
from pathlib import Path  # helps build file paths that work on Windows, Mac and Linux

import matplotlib  # the library that draws charts

matplotlib.use("Agg")  # draw charts straight into image files instead of opening a window
import matplotlib.pyplot as plt  # the part of matplotlib we use to make figures
import numpy as np  # maths on whole columns of numbers at once (e.g. logarithm)
import pandas as pd  # works with tables, like Excel inside Python
from sklearn.cluster import DBSCAN  # finds groups of rows that sit very close together
from sklearn.decomposition import PCA  # turns 5 columns into 2 so we can draw them flat
from sklearn.preprocessing import StandardScaler  # puts columns with different units on one common scale

# ---------------------------------------------------------------- where files are
HERE = Path(__file__).parent  # the folder this script is in (France/)
OUT = HERE / "out"  # a sub-folder for our results
OUT.mkdir(exist_ok=True)  # create that sub-folder if it is not there yet

df = pd.read_csv(HERE / "paris_listings.csv")  # load the spreadsheet into a table called df

# ---------------------------------------------------------------- step 1: prepare the 5 columns
# The columns have very different units: latitude moves in tiny decimals, availability
# goes from 0 to 365, price from 40 to 1500. If we compared them as they are, the
# biggest numbers would drown out the rest. So:
#   - price and reviews_per_month have a few huge values and many small ones, so we take
#     their logarithm, which shrinks the huge values and spreads out the small ones;
#   - then StandardScaler rescales every column so they all vary by the same amount.
X = StandardScaler().fit_transform(pd.DataFrame({
    "latitude": df.latitude,  # where the place is: north/south
    "longitude": df.longitude,  # where the place is: east/west
    "log_price": np.log(df.price),  # price per night (log-shrunk)
    "log_reviews_per_month": np.log(df.reviews_per_month),  # how often guests write reviews (log-shrunk)
    "availability_365": df.availability_365,  # how many nights are still free this year
}))
# X is now a table of numbers: one row per listing, 5 comparable columns.

# ---------------------------------------------------------------- step 2: find the tightly packed groups
# DBSCAN looks at every row and asks: "are there at least 5 other rows very close to me?"
# (eps=0.2 is how close "very close" means). Rows in crowded spots get joined into a
# group; rows in thin, spread-out areas get the label -1, meaning "not in any group".
labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)  # one group number per row; -1 = no group

sizes = pd.Series(labels[labels >= 0]).value_counts()  # how many rows each group has, biggest first
# The output looks like: 359, 354, 334, 319, 281, 272, 44, 32, ...
# The letters are the big groups. There is a sudden fall from 272 to 44, so we find the
# biggest fall (largest ratio between neighbours in that list) and keep everything before it.
n_letters = int(np.argmax(sizes.values[:-1] / sizes.values[1:])) + 1  # how many big groups come before the fall
letters = sizes.index[:n_letters]  # the group numbers of those big groups = the letters
print("cluster sizes:", sizes.head(10).tolist(), "->", n_letters, "letters")  # show it on screen

# ---------------------------------------------------------------- step 3: put the letters in order
# The story says the colours come from what a traveller weighs; we read the word from the
# cheapest letter to the most expensive one (average price of the rows in each letter).
order = df.groupby(labels).price.mean().loc[letters].sort_values().index  # letter groups, cheapest first

# ---------------------------------------------------------------- step 4: draw each letter
# Each letter lives in 5 columns, but a picture only has 2 directions (left-right, up-down).
# PCA finds the flat "sheet" the letter is drawn on and lays it on the page. The letter may
# come out turned or mirrored, because PCA does not know which way is "up".
fig, axs = plt.subplots(1, n_letters, figsize=(3 * n_letters, 3.5))  # one small picture per letter, in a row
for i, (ax, c) in enumerate(zip(axs, order)):  # i = position in the word (0,1,2..), c = group number
    p = PCA(2).fit_transform(X[labels == c])  # take only this letter's rows and flatten them to 2 columns
    ax.scatter(p[:, 0], p[:, 1], s=5, c="black")  # draw one black dot per row
    ax.set_aspect("equal")  # keep proportions so the letter is not squashed
    ax.set_xticks([]); ax.set_yticks([])  # hide axis numbers; they mean nothing here
    ax.set_title(f"#{i + 1}  cluster {c}  (n={(labels == c).sum()})", fontsize=9)  # position, group number, dot count
fig.suptitle("Letter clusters ordered by mean price per night")  # title over all pictures
fig.tight_layout()  # tidy spacing so titles do not overlap
fig.savefig(OUT / "letters.png", dpi=110)  # save the picture to out/letters.png

# ---------------------------------------------------------------- step 5: save which row is which letter
position = {c: i + 1 for i, c in enumerate(order)}  # group number -> place in the word (1 = first letter)
pd.DataFrame({
    "listing_id": df.listing_id,  # the id of every listing
    "cluster": [c if c in position else -1 for c in labels],  # its letter group, or -1 if it is a normal listing
    "position": [position.get(c, -1) for c in labels],  # which letter of the word it belongs to, -1 if none
}).to_csv(OUT / "labels.csv", index=False)  # save as out/labels.csv
print("saved", OUT / "letters.png", "and", OUT / "labels.csv")  # tell the user where the results are
