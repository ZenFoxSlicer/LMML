"""
Moldova - Feather Stories
=========================

THE PUZZLE IN PLAIN WORDS
-------------------------
Some birds wear a small ring on their leg with a 6-letter code on it.
Birdwatchers look at the ring through a telescope and write down the code.
They make two kinds of mistakes:

  1. A letter they cannot see at all is written as "?".
  2. A letter they think they see may be WRONG - and usually wrong in a
     predictable way: they mix up letters that LOOK alike from far away
     (for example O and D, Q and C, B and S).

We get two files:
  * observations.csv  - every bird sighting; some rows have a ring reading.
  * ring_register.csv - the official list of ringed birds and their true codes.

The register contains every ringed bird EXCEPT one White Stork.
Our job: figure out the true code on that missing stork's ring.

THE IDEA
--------
Imagine a pile of blurry photocopies of several documents. Most copies belong
to documents we already own an original of. To reconstruct the one document we
DON'T own, we must:

  a) learn what kind of blur the photocopier produces,
  b) throw away every copy we can explain as a blurry copy of an original,
  c) look at what is left, group copies that clearly belong together,
  d) combine them letter by letter ("majority vote") to undo the blur.

The puzzle also contains DECOYS: ring readings that appear again and again,
are not in the register, and look like "the missing bird" - but are really
just badly misread registered storks. Each step below defeats a kind of decoy.

Run:  python solve.py
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Windows consoles sometimes cannot print Romanian letters (ș, ă ...).
# This line makes printing safe everywhere.
sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
STORK = "Ciconia ciconia"  # scientific name of the White Stork


# ---------------------------------------------------------------------------
# STEP 0 - Load the data
# ---------------------------------------------------------------------------
# pandas reads a CSV file into a "DataFrame", which is just a table in memory.
obs = pd.read_csv(HERE / "observations.csv")
reg = pd.read_csv(HERE / "ring_register.csv")

# Only rows where somebody actually read a ring are useful to us.
reads = obs[obs["ring_read"].notna()].copy()

registered_rings = reg["ring"].tolist()                      # all true codes
registered_storks = reg.loc[reg["species"] == STORK, "ring"].tolist()
stork_reads = reads[reads["species"] == STORK].copy()

print(f"Sightings with a ring reading : {len(reads)}")
print(f"Registered ringed birds       : {len(reg)}")
print(f"Ring readings on White Storks : {len(stork_reads)}")
print(f"Registered White Storks       : {len(registered_storks)}")


# ---------------------------------------------------------------------------
# STEP 1 - The naive attempt (and why it fails)
# ---------------------------------------------------------------------------
# Simplest idea: "take stork readings that are not in the register, and pick
# the one that appears most often". Let's see what that gives.
print("\n=== STEP 1: naive count of unregistered stork readings ===")
unregistered = stork_reads[~stork_reads["ring_read"].isin(registered_rings)]
# Ignore readings with "?" here - they are incomplete.
complete = unregistered[~unregistered["ring_read"].str.contains("?", regex=False)]
top = complete["ring_read"].value_counts().head(5)
print(top.to_string())
print("-> Several codes are tied at the top. Counting alone cannot decide.")
print("   Some of these must be misreadings of birds that ARE registered.")


# ---------------------------------------------------------------------------
# STEP 2 - Measuring how different two codes are ("distance")
# ---------------------------------------------------------------------------
# To decide whether a reading is "a blurry copy" of a registered code, we
# need a number that says how far apart two codes are.
#
# The classic choice is the HAMMING DISTANCE: compare the codes position by
# position and count the positions where the letters differ.
#     RHGLWQ vs RHGLWD -> 1 difference -> distance 1
# A "?" means "unknown", so it can never count as a difference.

def plain_distance(read: str, true: str) -> int:
    """Count positions where a known letter disagrees with the true code."""
    return sum(1 for r, t in zip(read, true) if r != "?" and r != t)


# ---------------------------------------------------------------------------
# STEP 3 - Learn WHICH letters people confuse (the "error model")
# ---------------------------------------------------------------------------
# Instead of guessing which letters look alike, we let the data tell us.
#
# Take every reading that is almost certainly a copy of a known ring:
# its closest registered code is at most 1 letter away. In those readings
# the single wrong letter shows us one real mistake, e.g. true "O" written
# as "D". Counting these mistakes over thousands of sightings reveals the
# letter pairs that are confused again and again.
print("\n=== STEP 3: learning which letters observers confuse ===")
mistakes = Counter()
for read in reads["ring_read"]:
    distances = sorted((plain_distance(read, t), t) for t in registered_rings)
    best_d, best_ring = distances[0]
    second_d = distances[1][0]
    # Keep only confident matches: very close, and clearly closer than #2.
    if best_d <= 1 and second_d > best_d:
        for r, t in zip(read, best_ring):
            if r != "?" and r != t:
                # frozenset({"O","D"}) == frozenset({"D","O"}): the pair has no order
                mistakes[frozenset((r, t))] += 1

# A pair seen at least 5 times is a real, repeatable confusion, not chance.
LOOKALIKES = {pair for pair, n in mistakes.items() if n >= 5}
print("Look-alike pairs found:",
      " ".join("/".join(sorted(p)) for p in sorted(LOOKALIKES, key=sorted)))


# ---------------------------------------------------------------------------
# STEP 4 - A smarter distance that understands look-alike letters
# ---------------------------------------------------------------------------
# Now we upgrade the distance:
#   * "?"                    -> costs 0   (no information)
#   * same letter            -> costs 0
#   * look-alike swap (O/D)  -> costs 0.3 (a very common, forgivable mistake)
#   * any other wrong letter -> costs 1   (a "real" disagreement)
#
# With plain Hamming distance, a reading where EVERY letter was swapped for a
# look-alike would look 6 letters away from its true code - a total stranger.
# With this smarter distance it is only 6 x 0.3 = 1.8 away - clearly the same
# bird. This is what unmasks the hardest decoy.
LOOKALIKE_COST = 0.3


def smart_distance(read: str, true: str) -> float:
    cost = 0.0
    for r, t in zip(read, true):
        if r == "?" or r == t:
            continue
        cost += LOOKALIKE_COST if frozenset((r, t)) in LOOKALIKES else 1.0
    return cost


def closest_stork(read: str):
    """Return (smart distance, ring) of the registered stork nearest to read."""
    return min((smart_distance(read, t), t) for t in registered_storks)


# ---------------------------------------------------------------------------
# STEP 5 - Explain away the decoys
# ---------------------------------------------------------------------------
# For every stork reading, find the closest REGISTERED stork. If it is close
# enough (smart distance <= EXPLAINED), the reading is just a blurry copy of
# a bird we already know - so it cannot be the missing bird.
EXPLAINED = 2.0

stork_reads[["dist", "nearest"]] = stork_reads["ring_read"].apply(
    lambda r: pd.Series(closest_stork(r)))

print("\n=== STEP 5: what the 'top suspects' from step 1 really are ===")
for code in top.index:
    d_plain = min(plain_distance(code, t) for t in registered_storks)
    d_smart, ring = closest_stork(code)
    verdict = "misread of registered stork " + ring if d_smart <= EXPLAINED \
        else "NOT explained by any registered stork"
    print(f"{code}: plain distance {d_plain}, smart distance {d_smart:.1f}"
          f" -> {verdict}")

unexplained = stork_reads[stork_reads["dist"] > EXPLAINED]
print(f"\nStork readings explained by known birds : "
      f"{(stork_reads['dist'] <= EXPLAINED).sum()}")
print(f"Stork readings left unexplained         : {len(unexplained)}")


# ---------------------------------------------------------------------------
# STEP 6 - Group what is left by PLACE
# ---------------------------------------------------------------------------
# White storks are faithful to their nest: they come back to the same spot
# every spring. So readings of the missing stork should pile up at ONE place,
# while leftover noise is scattered around the country.
print("\n=== STEP 6: where do the unexplained stork readings come from? ===")
by_place = unexplained.groupby(["locality_id", "district"]).size() \
    .sort_values(ascending=False)
print(by_place.head(5).to_string())

home_locality = by_place.index[0][0]
home = unexplained[unexplained["locality_id"] == home_locality] \
    .sort_values("date")
lat, lon = home["latitude"].iloc[0], home["longitude"].iloc[0]
print(f"\nHome of the mystery stork: {home_locality} ({lat:.3f}N, {lon:.3f}E)")
print("That is the Lower Prut floodplain (lakes Manta and Beleu), Cahul -"
      " the reserve from the story.")
print(f"Seen there {len(home)} times, {home['date'].min()[:4]}"
      f" to {home['date'].max()[:4]}:")
print(" ".join(home["ring_read"]))


# ---------------------------------------------------------------------------
# STEP 7 - Majority vote, one letter position at a time ("consensus")
# ---------------------------------------------------------------------------
# Each reading is a noisy copy. For position 1 we collect the first letter of
# every reading (skipping "?") and pick the most common one; then position 2,
# and so on. Individual mistakes are outvoted by the many correct readings -
# the same reason averaging many measurements gives a more accurate result.
print("\n=== STEP 7: letter-by-letter majority vote ===")
answer = ""
for pos in range(6):
    letters = Counter(r[pos] for r in home["ring_read"] if r[pos] != "?")
    winner, votes = letters.most_common(1)[0]
    answer += winner
    print(f"position {pos + 1}: {dict(letters.most_common())}"
          f" -> '{winner}' ({votes}/{sum(letters.values())} votes)")

# Final sanity checks: the answer must be new, and not a disguised known bird.
assert answer not in registered_rings, "answer is already in the register!"
d, ring = closest_stork(answer)
print(f"\nClosest registered stork to {answer}: {ring}"
      f" (smart distance {d:.1f}) -> a genuinely new bird.")

print(f"\nFLAG: SIGMOID_{answer}")
