"""
filter_compare.py

Alfred 2.0 - compare filter models

Answers the question "is model X any good at filtering my memories?"
against the REAL database instead of against sample data - and does so
for several models side by side, so the selections can be compared
directly.

Usage in the project directory:

    python filter_compare.py "Where do I work?"
    python filter_compare.py                     (asks interactively)
    python filter_compare.py --list              (only lists all memories)

Enter the models to be compared in MODELLE below.

Touches the database read-only. The web clients can keep running in
parallel.
"""

import sys

from config_guard import require_config
require_config()   # readable message if config.py is missing

import config
from config import USER_ID

# Read with getattr so an older config.py without this setting keeps
# working - it then uses the name the project has always used.
CHROMA_COLLECTION = getattr(config, "CHROMA_COLLECTION", "alfred_memory")
from memory_filter_cloud import CloudMemoryFilter

# --------------------------------------------------------
# Which models should compete against each other?
# --------------------------------------------------------
# This is the place to answer "does Gemini Flash 2.5 work for
# filtering?" - just comment lines in and out.

MODELLE = [
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "~anthropic/claude-haiku-latest",
]

# Set to True to additionally see the complete list of memories.
ALLE_ANZEIGEN = False


# --------------------------------------------------------
# Load memories (read-only, own access just like in the web client)
# --------------------------------------------------------

def lade_erinnerungen(user_id):
    from datetime import datetime

    import chromadb

    from config import CHROMA_DB_DIR

    collection_name = CHROMA_COLLECTION
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(collection_name)

    raw = collection.get(
        where={"user_id": user_id},
        include=["metadatas", "documents"],
    )

    eintraege = []
    for md, doc in zip(raw.get("metadatas") or [], raw.get("documents") or []):
        md = md or {}
        text = md.get("data") or doc or ""
        if not text:
            continue

        created = md.get("created_at")
        datum = ""
        if created:
            try:
                datum = datetime.fromisoformat(
                    str(created).replace("Z", "+00:00")
                ).strftime("%d.%m.%Y") + ": "
            except Exception:
                datum = ""

        eintraege.append((created or "", f"{datum}{text}"))

    eintraege.sort(key=lambda p: p[0], reverse=True)

    gesehen = set()
    ergebnis = []
    for _, anzeige in eintraege:
        if anzeige not in gesehen:
            gesehen.add(anzeige)
            ergebnis.append(anzeige)

    return ergebnis


# --------------------------------------------------------
# Test a single model
# --------------------------------------------------------

def teste_modell(modell, frage, erinnerungen):
    """Returns a result dict - errors are not raised, so that a broken
    model does not abort the comparison."""

    filt = CloudMemoryFilter(model=modell)

    try:
        auswahl = filt.select_memories(user_prompt=frage, memories=erinnerungen)
    except Exception as e:
        return {"modell": modell, "fehler": str(e)}

    usage = filt.last_usage or {}

    return {
        "modell": modell,
        "auswahl": auswahl,
        "dauer": usage.get("elapsed"),
        "kosten": usage.get("cost"),
        "token": usage.get("total_tokens"),
    }


def kuerze(text, laenge=78):
    return text if len(text) <= laenge else text[:laenge - 1] + "…"


# --------------------------------------------------------
# Main program
# --------------------------------------------------------

def main():
    argumente = [a for a in sys.argv[1:]]
    nur_liste = "--list" in argumente
    argumente = [a for a in argumente if not a.startswith("--")]

    print("Loading memories from the database ...")
    erinnerungen = lade_erinnerungen(USER_ID)
    print(f"{len(erinnerungen)} memory/memories found for user_id '{USER_ID}'.\n")

    if not erinnerungen:
        print("No memories present - nothing to filter.")
        return

    if nur_liste or ALLE_ANZEIGEN:
        print("All memories:")
        for i, m in enumerate(erinnerungen):
            print(f"  {i:3d}  {kuerze(m)}")
        print()
        if nur_liste:
            return

    frage = " ".join(argumente).strip()
    if not frage:
        frage = input("Question for Alfred: ").strip()
    if not frage:
        return

    print(f"\nQuestion: {frage}")
    print("=" * 80)

    ergebnisse = []

    for modell in MODELLE:
        print(f"\n{modell}")
        print("-" * len(modell))

        ergebnis = teste_modell(modell, frage, erinnerungen)
        ergebnisse.append(ergebnis)

        if "fehler" in ergebnis:
            print(f"  ERROR: {ergebnis['fehler']}")
            continue

        auswahl = ergebnis["auswahl"]
        print(f"  {len(auswahl)} of {len(erinnerungen)} selected", end="")

        if ergebnis["dauer"] is not None:
            print(f"  |  {ergebnis['dauer']:.2f}s", end="")
        if ergebnis["token"]:
            print(f"  |  {ergebnis['token']} tokens", end="")
        if ergebnis["kosten"] is not None:
            print(f"  |  ${float(ergebnis['kosten']):.6f}", end="")
        print()

        if auswahl:
            for m in auswahl:
                print(f"    + {kuerze(m, 74)}")
        else:
            print("    (nothing selected - in the web client EVERYTHING")
            print("     would then be sent along to be safe)")

    # ----------------------------------------------------
    # Side-by-side comparison
    # ----------------------------------------------------

    gueltig = [e for e in ergebnisse if "fehler" not in e]

    if len(gueltig) < 2:
        return

    print("\n" + "=" * 80)
    print("Where do the models disagree?")
    print("=" * 80)

    mengen = {e["modell"]: set(e["auswahl"]) for e in gueltig}
    einig = set.intersection(*mengen.values())
    irgendwer = set.union(*mengen.values())
    strittig = irgendwer - einig

    print(f"\n  Chosen by all {len(gueltig)} models : {len(einig)}")
    print(f"  Chosen by only some     : {len(strittig)}")
    print(f"  Chosen by none          : {len(erinnerungen) - len(irgendwer)}")

    if strittig:
        print("\n  Disputed memories (X = chosen):\n")

        kopf = "  " + "".join(f"{i + 1:^4}" for i in range(len(gueltig)))
        print(kopf + "  Memory")
        print("  " + "-" * (4 * len(gueltig) + 2 + 60))

        for m in sorted(strittig):
            marker = "".join(
                f"{'X' if m in mengen[e['modell']] else '·':^4}" for e in gueltig
            )
            print(f"  {marker}  {kuerze(m, 60)}")

        print("\n  Legend:")
        for i, e in enumerate(gueltig, 1):
            print(f"    {i} = {e['modell']}")

    print(
        "\nNote: the 'right' selection is the one that contains everything Alfred\n"
        "needs in order to answer this question. A model that consistently picks\n"
        "fewer memories than the others is not being economical - it is forgetting."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
