"""Finland: Pipes, Leaks, and the Graph Beneath.

Finds the busiest towns in a route log.

General approach (does not depend on any particular input file):
  1. Clean every town name the same way (spaces, punctuation, capitals, accents).
  2. Translate Swedish town names to Finnish using a general table of Finland's
     bilingual municipalities (written from public knowledge, not from the data).
  3. Build the network: each pair of connected towns counts once, whatever the
     direction, route type, office or date.
  4. Rank towns by how many different towns they connect to.

Usage:  python solve.py [path/to/routes.csv]
"""
import argparse  # Built-in tool to read options typed after "python solve.py" on the command line.
import re  # Built-in tool for text patterns ("regular expressions"), used here to tidy spacing and punctuation.
import unicodedata  # Built-in tool that knows about letters and accents (e.g. that "ä" is "a" + two dots).
from collections import defaultdict  # A dictionary that creates an empty entry the first time you use a new key.

import pandas as pd  # A popular library for reading tables (CSV files) into Python.

# Finland has two official languages, so many towns have a Swedish name as well as a Finnish one.
# This table covers Finnish municipalities whose Swedish name differs: Swedish name -> Finnish name.
# Names are written with their real spelling; the program cleans them the same way as the data before use.
SWEDISH_TO_FINNISH = {
    "Helsingfors": "Helsinki", "Esbo": "Espoo", "Vanda": "Vantaa", "Grankulla": "Kauniainen",
    "Åbo": "Turku", "Tammerfors": "Tampere", "Uleåborg": "Oulu", "Vasa": "Vaasa",
    "Björneborg": "Pori", "Lahtis": "Lahti", "Villmanstrand": "Lappeenranta",
    "Tavastehus": "Hämeenlinna", "S:t Michel": "Mikkeli", "Nyslott": "Savonlinna",
    "Borgå": "Porvoo", "Karleby": "Kokkola", "Jakobstad": "Pietarsaari", "Brahestad": "Raahe",
    "Fredrikshamn": "Hamina", "Lovisa": "Loviisa", "Hangö": "Hanko", "Raseborg": "Raasepori",
    "Ekenäs": "Tammisaari", "Lojo": "Lohja", "Kyrkslätt": "Kirkkonummi", "Sibbo": "Sipoo",
    "Sjundeå": "Siuntio", "Ingå": "Inkoo", "Vichtis": "Vihti", "Tusby": "Tuusula",
    "Träskända": "Järvenpää", "Kervo": "Kerava", "Hyvinge": "Hyvinkää", "Högfors": "Karkkila",
    "Borgnäs": "Pornainen", "Mörskom": "Myrskylä", "Lappträsk": "Lapinjärvi",
    "Pyttis": "Pyhtää", "Itis": "Iitti", "Vederlax": "Virolahti",
    "Idensalmi": "Iisalmi", "Kajana": "Kajaani", "Torneå": "Tornio", "Övertorneå": "Ylitornio",
    "S:t Karins": "Kaarina", "Reso": "Raisio", "Nådendal": "Naantali", "Nystad": "Uusikaupunki",
    "Raumo": "Rauma", "Kumo": "Kokemäki", "Ulvsby": "Ulvila", "Ikalis": "Ikaalinen",
    "Birkala": "Pirkkala", "Tavastkyro": "Hämeenkyrö", "Virdois": "Virrat", "Vittis": "Huittinen",
    "Virmo": "Mynämäki", "Lundo": "Lieto", "Pargas": "Parainen", "Kimitoön": "Kemiönsaari",
    "Pemar": "Paimio", "Nousis": "Nousiainen", "Tövsala": "Taivassalo", "Gustavs": "Kustavi",
    "Letala": "Laitila", "Euraåminne": "Eurajoki", "Jockis": "Jokioinen",
    "Ilmola": "Ilmajoki", "Lappo": "Lapua", "Alavo": "Alavus", "Kristinestad": "Kristiinankaupunki",
    "Kaskö": "Kaskinen", "Närpes": "Närpiö", "Malax": "Maalahti", "Korsholm": "Mustasaari",
    "Vörå": "Vöyri", "Nykarleby": "Uusikaarlepyy", "Larsmo": "Luoto", "Kronoby": "Kruunupyy",
    "Kaustby": "Kaustinen", "Östermark": "Teuva", "Storkyro": "Isokyrö", "Laihela": "Laihia",
    "Mariehamn": "Maarianhamina", "Enare": "Inari", "Enontekis": "Enontekiö", "Utsjok": "Utsjoki",
    "Keuru": "Keuruu", "Limingo": "Liminka", "Karlö": "Hailuoto", "Ijo": "Ii",
}


def fold(name: str) -> str:  # Turn any messy spelling of a town name into one tidy, comparable form.
    s = unicodedata.normalize("NFKD", str(name))  # Split accented letters into "plain letter + accent mark" ("ä" -> "a" + "¨").
    s = "".join(c for c in s if not unicodedata.combining(c))  # Drop the accent marks, keeping only the plain letters.
    s = s.lower()  # Make all letters small, so "ORIVESI" and "Orivesi" match.
    s = re.sub(r"[^\w:\s-]", " ", s)  # Replace stray punctuation (dots, commas, quotes...) with spaces; keep letters, ":" (S:t) and "-".
    s = re.sub(r"\s*-\s*", "-", s)  # Remove spaces around hyphens, so "Mänttä - Vilppula" becomes "mantta-vilppula".
    return re.sub(r"\s+", " ", s).strip()  # Squeeze repeated spaces into one and trim spaces at both ends.


# Build the lookup table actually used by the program: both sides cleaned with fold().
ALIASES = {fold(sv): fold(fi) for sv, fi in SWEDISH_TO_FINNISH.items()}


def canon(name: str) -> str:  # Give each town its single official ("canonical") name.
    f = fold(name)  # First tidy the spelling.
    return ALIASES.get(f, f)  # If it is a known Swedish name, swap in the Finnish one; otherwise keep it.


def load_routes(path: str) -> pd.DataFrame:  # Read the log file and return a table with two clean columns: a and b.
    df = pd.read_csv(path, dtype=str, encoding="utf-8")  # Load the CSV; treat every value as text.
    df = df.dropna(subset=["from_town", "to_town"])  # Drop rows where either end of the route is missing.
    df["a"] = df["from_town"].map(canon)  # New column "a": the cleaned starting town of each route.
    df["b"] = df["to_town"].map(canon)  # New column "b": the cleaned destination town of each route.
    return df[(df["a"] != "") & (df["b"] != "")]  # Keep only rows where both names are non-empty after cleaning.


def build_network(df: pd.DataFrame):  # Turn the list of routes into a network of towns and their connections.
    # frozenset((a, b)) is an unordered pair, so A->B and B->A are the same connection.
    # The outer { ... } is a set, which cannot hold duplicates, so repeated log entries collapse into one.
    # "if a != b" skips routes that start and end in the same town.
    edges = {frozenset((a, b)) for a, b in zip(df["a"], df["b"]) if a != b}
    neighbours = defaultdict(set)  # For each town, the set of towns it is directly connected to.
    for edge in edges:  # Go through every unique connection.
        a, b = tuple(edge)  # Take the two towns out of the pair.
        neighbours[a].add(b)  # b is a neighbour of a...
        neighbours[b].add(a)  # ...and a is a neighbour of b.
    towns = set(df["a"]) | set(df["b"])  # Every distinct town in the log ("|" joins two sets).
    return towns, edges, neighbours  # Hand all three back to whoever called this function.


def rank(neighbours) -> list:  # Sort towns from most connections to fewest.
    # "-len(...)" puts the biggest counts first; the town name breaks ties so the order is always the same.
    return sorted(((t, len(n)) for t, n in neighbours.items()), key=lambda x: (-x[1], x[0]))


def main():  # The main program, run step by step.
    parser = argparse.ArgumentParser(description="Rank towns in a route log by distinct connections.")  # Set up command-line options.
    parser.add_argument("path", nargs="?", default="routes.csv", help="route log CSV (default: routes.csv)")  # Optional file path.
    parser.add_argument("--top", type=int, default=7, help="how many towns form the flag word (default: 7)")  # Optional flag length.
    args = parser.parse_args()  # Read what the user actually typed.

    df = load_routes(args.path)  # Steps 1-2: load the log, clean names, translate Swedish names.
    towns, edges, neighbours = build_network(df)  # Step 3: build the network.
    ranking = rank(neighbours)  # Step 4: rank the towns.

    print(f"log rows: {len(df)}  towns: {len(towns)}  connections: {len(edges)}")  # Totals, to compare with the task's check (116 / 448).
    for town, degree in ranking[:10]:  # Show the 10 busiest towns...
        print(f"  {town:20s} {degree}")  # ...with how many towns each connects to.

    top = ranking[: args.top]  # The towns that make up the flag word.
    if len(ranking) > args.top and ranking[args.top - 1][1] == ranking[args.top][1]:  # Is the last place in the word shared with the next town?
        print("warning: tie at the cut-off, the flag word may be ambiguous")  # Then the answer would depend on the alphabetical tie-break.
    word = "".join(town[0] for town, _ in top).upper()  # First letter of each top town, joined and in capitals.
    print(f"FLAG: SIGMOID_{word}")  # Final answer in the required format.


if __name__ == "__main__":  # True only when the file is run directly (python solve.py), not imported.
    main()  # Start the program.
