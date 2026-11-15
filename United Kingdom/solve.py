"""
Call My BLEUff  -  United Kingdom
=================================
Run:  python solve.py      (needs pandas, matplotlib; bleu.py in the same folder)

THE GAME IN PLAIN WORDS
-----------------------
24 rounds. In each round three panellists ("seats" 1, 2, 3) each write one short
description for the same 40 venues (pubs, coffee shops, restaurants).
For every venue the guide's own writers wrote 5 "reference" descriptions.
One panellist per round is honest; the other two "bluff" on some cards.

THE JUDGE: BLEU (bleu.py)
-------------------------
BLEU does not understand language. It only counts overlapping word chunks:
  * it cuts a text into 1-, 2-, 3- and 4-word chunks ("n-grams");
  * PRECISION = what share of those chunks also appear in some reference;
  * the four precisions are multiplied together (geometric mean);
  * BREVITY PENALTY: a deck shorter than the references is punished. This is
    computed ONCE for the whole deck, so long cards can pay for short ones.
BLEU only asks "is what you said found in the references?" (precision).
It never asks "did you say everything that matters?" (recall), and it
never notices nonsense or repetition as long as the chunks match.

HOW THE BLUFFERS GAME IT (found by reading their cards)
-------------------------------------------------------
1. HOLLOW CARDS - say only the safest words and drop the facts:
     "The Wrestlers is a family friendly coffee shop."   (price? food? area?)
   Every chunk matches, so precision is high.
2. SNIPPETS     - a piece cut straight out of a reference: "Customer rating 1 out."
3. PADDING      - restate the same facts again and again ("more than more than",
   "It is kid-friendly and moderately priced. ... a moderately priced kids
   friendly pub"). This makes the deck long, so the brevity penalty caused by the
   short cards disappears, and repeated reference phrases still match.
4. WORD SALAD   - scraps of different references glued together:
     "Zizzi is an with Indian food, has a pub with is highly rated ..."

OUR METRIC: "THE LISTENER'S CHECKLIST"
--------------------------------------
A listener wants two things from a description:
  (a) all the facts about the venue  -> recall, which BLEU ignores
  (b) each fact once                 -> no padding
Each venue has a short list of facts: type of place, food, price, customer
rating, area, family-friendliness. We know a fact belongs to a venue when
most (3+) of its 5 references mention it. A card is GOOD if it mentions every
such fact and does not repeat itself. Deck score = number of good cards (0-40).
Hollow cards, snippets and word salad fail (a), padding fails (b).
The honest panellist = the deck with the most good cards in its round.
"""
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bleu import corpus_bleu, segment_stats, tokenize

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 200)

# =============================================================== 1. load data
refs = pd.read_csv(HERE / "references.csv")
panel = pd.read_csv(HERE / "panel.csv")
# (round, card) -> list of its 5 reference texts
ref_map = refs.groupby(["round", "card"]).reference.apply(list).to_dict()
print(f"{len(refs)} reference texts, {len(panel)} panel cards\n")

# ============================================== 2. the fact checklist
# For each kind of fact: a "regular expression" = a list of words/phrases,
# separated by |, any of which counts as mentioning that fact.
# (\b means "word boundary", so "pub" does not match inside "public".)
FACTS = {
    "type":   r"\bpub\b|coffee ?shop|coffee house|restaurant|eatery|bistro|\bplace\b"
              r"|venue|establishment|\bjoint\b|\bchain\b",
    "food":   r"french|italian|english|british|japanese|sushi|chinese|indian"
              r"|fast[ -]?food|burger|fries|takeaway|oriental|asian|curry|pasta|wine",
    "price":  r"cheap|inexpensive|afford|low[- ]?(cost|price|priced)|less than|under"
              r"|below|budget|moderate|average[- ]?(price|priced)|mid|20-25|20 to 25"
              r"|reasonabl|expensive|pricey|high[- ]?(end|price|priced|cost)|more than"
              r"|over|30|upmarket|premium|costly|price",
    "rating": r"rat(ed|ing|ings)|star|review|\d out of 5|out of five|score",
    "area":   r"riverside|river|city cent(re|er)|centre|center|waterfront",
    "family": r"famil|kid|child|adult",
}


def facts_in(text):
    """Which kinds of fact does this text mention?"""
    t = text.lower()
    return {name for name, pattern in FACTS.items() if re.search(pattern, t)}


def venue_facts(references):
    """The facts most references (3 or more of 5) agree on."""
    votes = Counter(f for r in references for f in facts_in(r))
    return {f for f, n in votes.items() if n >= 3}


# Small "glue" words that are naturally repeated; we ignore them when looking
# for padding.
GLUE = set("a an the is it its it's of in and to with for on at by near that this"
           " there which has have are be as from they their also".split())


def repeated_words(text):
    """How many times a meaningful word appears again in the same card.
    Normal writing: 0 or 1 (the venue name twice, say). Padding: 2 and more."""
    words = [w for w in tokenize(text) if w.isalnum() and w not in GLUE]
    return sum(n - 1 for n in Counter(words).values() if n > 1)


# ============================================ 3. score every card of the panel
rows = []
for r in panel.itertuples():
    five = ref_map[(r.round, r.card)]
    missing = venue_facts(five) - facts_in(r.text)
    st = segment_stats(r.text, five)
    rows.append({
        "round": r.round, "card": r.card, "seat": r.seat,
        "missing": len(missing),                # facts the listener did not get
        "padded": repeated_words(r.text) >= 2,  # says the same thing again
        # for the evidence table: BLEU's own 4-word precision of this card
        "p4": st["matched"][3] / st["total"][3] if st["total"][3] else 0.0,
        "length_ratio": st["out_len"] / st["ref_len"],
    })
cards = pd.DataFrame(rows)
cards["hollow"] = cards.missing >= 1
cards["good"] = ~cards.hollow & ~cards.padded

# ============================================== 4. one line per deck (seat)
deck_rows = []
for (rnd, seat), deck in panel.groupby(["round", "seat"]):
    res = corpus_bleu(list(deck.text), [ref_map[(rnd, c)] for c in deck.card])
    sub = cards[(cards["round"] == rnd) & (cards.seat == seat)]
    deck_rows.append({
        "round": rnd, "seat": seat,
        "BLEU": round(res["bleu"], 3),
        "brevity_pen": round(res["brevity_penalty"], 3),
        "4gram_prec": round(res["precisions"][3], 3),
        "shortest_card": round(sub.length_ratio.min(), 2),
        "longest_card": round(sub.length_ratio.max(), 2),
        "hollow_cards": int(sub.hollow.sum()),
        "padded_cards": int(sub.padded.sum()),
        "GOOD_cards": int(sub.good.sum()),
    })
decks = pd.DataFrame(deck_rows)
print("Per deck: the house BLEU and its ingredients, then our checklist")
print("(length = card length / closest reference length; good = all facts, no padding)")
print(decks.to_string(index=False))


# ================================================== 5. verdicts and the flag
def decode(seats):
    """Seats in round order -> word. 3 rounds per letter, base 3, A=0 ... Z=25."""
    word = ""
    for i in range(0, len(seats), 3):
        d1, d2, d3 = (s - 1 for s in seats[i:i + 3])
        word += chr(ord("A") + 9 * d1 + 3 * d2 + d3)
    return word


verdict = []
for rnd, g in decks.groupby("round"):
    g = g.sort_values("GOOD_cards", ascending=False)
    verdict.append({
        "round": rnd,
        "BLEU_winner": int(g.loc[g.BLEU.idxmax(), "seat"]),
        "honest": int(g.seat.iloc[0]),
        "good cards (honest / others)":
            f"{g.GOOD_cards.iloc[0]} / {g.GOOD_cards.iloc[1]}, {g.GOOD_cards.iloc[2]}",
        "margin": int(g.GOOD_cards.iloc[0] - g.GOOD_cards.iloc[1]),
    })
verdict = pd.DataFrame(verdict)
print("\nRound verdicts (margin = good cards of the honest deck minus the runner-up)")
print(verdict.to_string(index=False))

honest = decks.merge(verdict[["round", "honest"]])
honest = honest[honest.seat == honest.honest]
bluff = decks.merge(verdict[["round", "honest"]])
bluff = bluff[bluff.seat != bluff.honest]
print(f"\nHonest decks : {honest.GOOD_cards.min()}-{honest.GOOD_cards.max()} good cards of 40,"
      f" mean BLEU {honest.BLEU.mean():.3f}")
print(f"Bluffer decks: {bluff.GOOD_cards.min()}-{bluff.GOOD_cards.max()} good cards of 40,"
      f" mean BLEU {bluff.BLEU.mean():.3f}")
print("Smallest margin in any round:", verdict.margin.min(), "cards")
print("BLEU crowned the honest panellist in",
      int((verdict.BLEU_winner == verdict.honest).sum()), "of 24 rounds")

print("\nBLEU's choice (producers): SIGMOID_" + decode(list(verdict.BLEU_winner)))
print("FLAG (honest panellists) : SIGMOID_" + decode(list(verdict.honest)))

# ================================================================ 6. figure
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for seat, grp in decks.groupby("seat"):
    axes[0].scatter(grp["round"], grp.BLEU, label=f"seat {seat}", s=30)
    axes[1].scatter(grp["round"], grp.GOOD_cards, label=f"seat {seat}", s=30)
for ax, col in zip(axes, ["BLEU", "GOOD_cards"]):
    ax.scatter(honest["round"], honest[col], s=160, facecolors="none",
               edgecolors="black", label="honest")
    ax.legend(frameon=False, ncol=4, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("house BLEU")
axes[0].set_title("BLEU cannot tell the honest panellist apart ...")
axes[1].set_ylabel("good cards (of 40)")
axes[1].set_title("... the listener's checklist puts the honest deck on top in every round")
axes[1].set_xlabel("round")
axes[1].set_xticks(range(1, 25))
fig.tight_layout()
fig.savefig(HERE / "bleu_vs_checklist.png", dpi=130)
print("figure saved: bleu_vs_checklist.png")
