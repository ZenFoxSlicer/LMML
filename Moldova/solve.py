"""
Moldova - Feather Stories
=========================

THE PUZZLE IN PLAIN WORDS
-------------------------
Some birds wear a leg ring with a 6-letter code. Birdwatchers read it through
a telescope and write down what they saw:
  * a letter they could not see at all is written as "?";
  * a letter they DID see may still be wrong.

ring_register.csv lists the true code of every ringed bird except ONE White
Stork. We must work out the code on that stork's ring.

WHAT THE DATA TURNED OUT TO HIDE (discovered step by step, see below)
--------------------------------------------------------------------
1. The 24 usable letters fall into 6 "look-alike families" of 4 letters:
       A K R X | B G Z S | C O D Q | E F L P | H W N M | I Y J T
   (U and V never appear). A misread ALWAYS stays inside the family:
   an "O" may be written as C, D or Q, but never as, say, "E".

2. Inside a family the 4 letters behave like the numbers 0,1,2,3, and a
   misread is a fixed "shift" of that number (technically an XOR):
       shift 1 swaps  A<->K, R<->X,  B<->G, Z<->S, ...
       shift 2 swaps  A<->R, K<->X,  B<->Z, G<->S, ...
       shift 3 swaps  A<->X, K<->R,  B<->S, G<->Z, ...
   This is like reading a word through a slightly distorting lens: each
   letter is replaced by a specific look-alike, the SAME way every time.

3. The "lens" belongs to the PLACE where a bird lives (its home site, e.g.
   a nest or a colony seen from one fixed viewpoint). Every bird read at
   that site gets the same shift. Example: at site L10196035 the true ring
   FAGQKC is read as EKBDAO, and LLAOAO is read as PPKCKC - shift 1 for
   every letter of every bird there.

So a straightforward majority vote over the readings gives the code AS SEEN
THROUGH THE SITE'S LENS (RHGLWQ for our stork) - not the code on the ring.
That is why the obvious answer is wrong. To get the true code we must
"undo the lens", which means figuring out which shift the stork's home site
uses. Registered birds let us measure the lens at every other home site;
the mystery stork's site (Lower Prut, Cahul) has no registered bird, so the
data narrows it to four candidates (step 6). Which one is right is NOT
decided by anything this script found; RHGLWQ (shift 0) was rejected and
ANSEMO (shift 2) was confirmed by submitting them on the platform.

Run:  python solve.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # safe printing of Romanian names
HERE = Path(__file__).parent
STORK = "Ciconia ciconia"

# The six look-alike families. A letter's position inside its string is its
# "number" 0..3, chosen so that every observed misread is an XOR shift.
FAMILIES = ["AKRX", "BGZS", "CODQ", "EFLP", "HWNM", "IYJT"]
FAMILY_OF = {ch: f for f, fam in enumerate(FAMILIES) for ch in fam}
NUMBER_OF = {ch: n for fam in FAMILIES for n, ch in enumerate(fam)}


def shift_letter(ch: str, s: int) -> str:
    """Apply lens shift s (0..3) to one letter, staying in its family."""
    if ch == "?":
        return ch
    fam = FAMILIES[FAMILY_OF[ch]]
    return fam[NUMBER_OF[ch] ^ s]


def shift_word(word: str, s: int) -> str:
    return "".join(shift_letter(c, s) for c in word)


def family_pattern(word: str):
    """The sequence of families, e.g. RHGLWQ -> (0,4,1,3,4,2). '?' -> None.
    Because misreads never leave a family, this pattern survives any
    misreading - it is the bird's unchangeable 'fingerprint'."""
    return tuple(None if c == "?" else FAMILY_OF[c] for c in word)


def fits(read: str, ring: str) -> bool:
    """Could this reading be a (misread) sighting of that ring?"""
    return all(a is None or a == b
               for a, b in zip(family_pattern(read), family_pattern(ring)))


# ---------------------------------------------------------------------------
# STEP 0 - Load
# ---------------------------------------------------------------------------
obs = pd.read_csv(HERE / "observations.csv")
reg = pd.read_csv(HERE / "ring_register.csv")
reads = obs[obs["ring_read"].notna()].copy()
print(f"Ring readings: {len(reads)}   Registered birds: {len(reg)}")


# ---------------------------------------------------------------------------
# STEP 1 - Why the naive answer is a trap
# ---------------------------------------------------------------------------
# Most readings are wrong in at least one letter, so "find the code that is
# not in the register" produces many fake candidates.
complete = reads[~reads["ring_read"].str.contains("?", regex=False)]
exact = complete["ring_read"].isin(reg["ring"]).mean()
print(f"\n=== STEP 1 ===\nFully readable sightings that match the register "
      f"exactly: {exact:.0%}. Most readings contain mistakes.")


# ---------------------------------------------------------------------------
# STEP 2 - Check the look-alike families on the data
# ---------------------------------------------------------------------------
# Link every reading to a registered bird of the same species whose
# fingerprint (family pattern) fits. If exactly one bird fits, we know who
# was seen.
rings_by_species = defaultdict(list)
for ring, sp in zip(reg["ring"], reg["species"]):
    rings_by_species[sp].append(ring)


def identify(read, species):
    cands = [r for r in rings_by_species[species] if fits(read, r)]
    return cands[0] if len(cands) == 1 else None


reads["bird"] = [identify(q, s) for q, s in zip(reads["ring_read"],
                                                  reads["species"])]
known = reads[reads["bird"].notna()]
same_family = sum(FAMILY_OF[p] == FAMILY_OF[t]
                  for q, t in zip(known["ring_read"], known["bird"])
                  for p, t in zip(q, t) if p != "?")
letters = sum(p != "?" for q in known["ring_read"] for p in q)
print(f"\n=== STEP 2 ===\nReadings identified to one registered bird: "
      f"{len(known)}")
print(f"Read letters that stay inside their look-alike family: "
      f"{same_family}/{letters}  (100% means the families are right)")


# ---------------------------------------------------------------------------
# STEP 3 - Find the stork that no registered bird can explain
# ---------------------------------------------------------------------------
storks = reads[reads["species"] == STORK]
reg_storks = rings_by_species[STORK]
orphan = storks[[not any(fits(q, r) for r in reg_storks)
                 for q in storks["ring_read"]]]
print(f"\n=== STEP 3 ===\nStork readings whose fingerprint fits NO "
      f"registered stork: {len(orphan)}")
print(orphan.groupby(["locality_id", "district"]).size().to_string())
home_site = orphan["locality_id"].mode()[0]
lat, lon = orphan["latitude"].iloc[0], orphan["longitude"].iloc[0]
print(f"-> All at {home_site} ({lat:.3f}N {lon:.3f}E): the Lower Prut "
      f"floodplain, lakes Manta/Beleu - the reserve from the story.")
print(f"   Seen {len(orphan)} times, {orphan['date'].min()[:4]}-"
      f"{orphan['date'].max()[:4]}: the stork lives there.")


# ---------------------------------------------------------------------------
# STEP 4 - What the watchers at Cahul see (majority vote per letter)
# ---------------------------------------------------------------------------
apparent = ""
for i in range(6):
    votes = Counter(q[i] for q in orphan["ring_read"] if q[i] != "?")
    apparent += votes.most_common(1)[0][0]
print(f"\n=== STEP 4 ===\nLetter-by-letter majority at Cahul: {apparent}")
print("This is the ring AS SEEN through the site's 'lens'. It is not "
      "necessarily the true code.")


# ---------------------------------------------------------------------------
# STEP 5 - Measure the lens at every home site with registered birds
# ---------------------------------------------------------------------------
# A registered bird's home = its ringed coordinates. For readings made at
# that exact spot, compare each read letter with the true letter and record
# the shift (0 = read correctly).
reg["home"] = list(zip(reg["ringed_latitude"].round(3),
                       reg["ringed_longitude"].round(3)))
home_of = dict(zip(reg["ring"], reg["home"]))
known = known.assign(spot=list(zip(known["latitude"].round(3),
                                   known["longitude"].round(3))))
at_home = known[known["spot"] == known["bird"].map(home_of)]

shifts_by_site = defaultdict(Counter)
for site, q, t in zip(at_home["locality_id"], at_home["ring_read"],
                      at_home["bird"]):
    for p, c in zip(q, t):
        if p != "?":
            shifts_by_site[site][NUMBER_OF[p] ^ NUMBER_OF[c]] += 1

# Keep sites with enough evidence; their lens is the most common shift.
site_lens = {}
for site, cnt in shifts_by_site.items():
    if sum(cnt.values()) >= 20:
        site_lens[site] = cnt.most_common(1)[0][0]
lens_count = Counter(site_lens.values())
print(f"\n=== STEP 5 ===\nHome sites with a measured lens: {len(site_lens)}")
for s in range(4):
    print(f"  lens shift {s}: {lens_count[s]} sites")
print("Example: site L10196035 has shift 1 - true FAGQKC is read as "
      f"{shift_word('FAGQKC', 1)}, LLAOAO as {shift_word('LLAOAO', 1)}.")

# Stray mistakes: letters that disagree with what the site's lens predicts.
# At shift-0 (undistorted) sites, stray mistakes are almost always shift 3.
# At distorted sites they are spread evenly over the other shifts.
stray_at_undistorted = Counter()
for site, lens in site_lens.items():
    if lens == 0:
        for shift, n in shifts_by_site[site].items():
            if shift:
                stray_at_undistorted[shift] += n


# ---------------------------------------------------------------------------
# STEP 6 - Four possible true codes for the stork
# ---------------------------------------------------------------------------
# The Cahul lens is unknown, so the true code is the apparent code with one
# of the four shifts undone. (XOR is its own inverse: undoing = re-applying.)
candidates = {s: shift_word(apparent, s) for s in range(4)}
print("\n=== STEP 6 ===\nIf the Cahul lens is shift s, the true ring is:")
for s, w in candidates.items():
    print(f"  shift {s}: {w}")


# ---------------------------------------------------------------------------
# STEP 7 - What the data can and cannot decide
# ---------------------------------------------------------------------------
# A weak hint: at undistorted (shift-0) sites, stray mistakes (letters that
# disagree with what the lens predicts) are mostly "shift 3". At Cahul they
# are somewhat more spread out, which mildly suggests Cahul IS distorted.
stray = Counter()
for q in orphan["ring_read"]:
    for p, c in zip(q, apparent):
        if p != "?" and p != c:
            stray[NUMBER_OF[p] ^ NUMBER_OF[c]] += 1
share3_cahul = stray[3] / sum(stray.values())
share3_undist = stray_at_undistorted[3] / sum(stray_at_undistorted.values())
print("\n=== STEP 7 ===")
print(f"Share of stray mistakes that are shift 3: undistorted sites "
      f"{share3_undist:.0%}, Cahul {share3_cahul:.0%} "
      f"(only {sum(stray.values())} stray letters - a weak hint).")
print("The data narrows the ring to the 4 candidates above but does NOT")
print("pick one: we tested district, location, observers, ringing year and")
print("species, and none predicts a site's lens. A probability model gave")
print("unstable results (the winner changed with an arbitrary cut-off).")
print("Platform feedback settles it:")
print(f"  {candidates[0]} (shift 0, the plain majority vote) - rejected")
print(f"  {candidates[2]} (shift 2)                          - accepted")

# ---------------------------------------------------------------------------
# STEP 8 - Answer
# ---------------------------------------------------------------------------
# Of the two remaining candidates, shift 2 was confirmed on the platform.
answer = candidates[2]
assert answer not in set(reg["ring"])
print(f"\nConfirmed on the platform: shift 2 -> {answer}")
print(f"FLAG: SIGMOID_{answer}")
