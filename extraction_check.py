"""
extraction_check.py

Alfred 2.0 - does the fact extraction work in your language?

Runs the same handful of statements through the extractor once per
language and prints what came back, side by side. Useful in two
situations:

    * you set ASSISTANT_LANGUAGE to something other than English or
      German and want to see whether extraction holds up
    * extraction behaves differently than you expect and you want
      evidence rather than a hunch

This is a probe, not a test suite. For a real measurement there is
memory_extractor_regression_test_v5.py with 43 German cases and a
score. What this does is make an asymmetry visible quickly, with a
handful of model calls instead of 43.

The probes are deliberately matched: every line means the same thing in
every language, so anything that works in one and not in another is a
property of the language handling, not of the sentence.

Usage:

    python extraction_check.py                     English and German
    python extraction_check.py --name Tester       as a different speaker
    python extraction_check.py --lang en           one language only
    python extraction_check.py --lang German       the same, spelled out

Language codes and names both work, upper or lower case.

Costs a fraction of a cent per run with the cloud backend.
"""

import argparse
import sys

from config_guard import require_config
require_config()   # readable message if config.py is missing

import config
from memory_extractor_v5 import MemoryExtractor

USER_NAME = getattr(config, "USER_NAME", "") or "Tester"
BACKEND = getattr(config, "MEMORY_EXTRACTOR_BACKEND", "openrouter")


# Matched probes. The comment after each says what SHOULD happen, so a
# wrong result is recognisable without thinking it through again.
PROBEN = {
    "English": [
        ("My hobby is testing.",                    "store: hobby"),
        ("I work as a nurse.",                      "store: profession"),
        ("I like pizza.",                           "store: preference"),
        ("I have two sons.",                        "store: family"),
        ("Hello Alfred!",                           "store nothing: small talk"),
        ("What do you know about me?",              "store nothing: a question"),
        ("I am tired today.",                       "store nothing: temporary"),
        ("I live in Munich, you don't need to remember that.",
                                                    "store nothing: opt-out"),
    ],
    "German": [
        ("Mein Hobby ist Testen.",                  "store: hobby"),
        ("Ich arbeite als Krankenpfleger.",         "store: profession"),
        ("Ich mag Pizza.",                          "store: preference"),
        ("Ich habe zwei Söhne.",                    "store: family"),
        ("Hallo Alfred!",                           "store nothing: small talk"),
        ("Was weißt du über mich?",                 "store nothing: a question"),
        ("Ich bin heute müde.",                     "store nothing: temporary"),
        ("Ich wohne in München, das musst du dir nicht merken.",
                                                    "store nothing: opt-out"),
    ],
}


# "en" and "English" should both work - the probes are keyed by the name
# the prompt uses, but nobody wants to type that out.
KUERZEL = {"en": "English", "eng": "English",
           "de": "German", "ger": "German", "deu": "German"}


def finde_sprache(eingabe):
    """Maps a code or a name to a key of PROBEN, or returns None."""
    wort = (eingabe or "").strip().lower()
    if wort in KUERZEL:
        return KUERZEL[wort]
    for name in PROBEN:
        if name.lower() == wort:
            return name
    return None


def baue_extractor():
    if BACKEND == "openrouter":
        from memory_extractor_cloud import CloudMemoryExtractor
        return CloudMemoryExtractor()
    if BACKEND == "ollama":
        return MemoryExtractor(
            model=getattr(config, "MEMORY_EXTRACTOR_MODEL", "llama3.1:8b"),
            host=getattr(config, "OLLAMA_HOST", "http://localhost:11434"),
            num_thread=getattr(config, "OLLAMA_NUM_THREAD", None),
        )
    print("MEMORY_EXTRACTOR_BACKEND is 'off' - nothing to check.")
    raise SystemExit(1)


def pruefe(sprache, proben, name):
    """Runs the probes of one language and returns (hits, expected)."""
    import memory_extractor_v5 as mod
    vorher = mod.ASSISTANT_LANGUAGE_NAME
    mod.ASSISTANT_LANGUAGE_NAME = sprache   # output language for this block
    extractor = baue_extractor()

    print(f"\n{'=' * 74}\n{sprache}  (speaker: {name})\n{'=' * 74}")
    treffer = erwartet = 0
    try:
        for satz, erwartung in proben:
            soll_speichern = erwartung.startswith("store:")
            erwartet += 1
            try:
                memories = extractor.extract(satz, user_name=name)
            except Exception as e:
                print(f"  ?  {satz}\n     ERROR: {e}")
                continue

            hat_gespeichert = bool(memories)
            richtig = hat_gespeichert == soll_speichern
            treffer += richtig

            print(f"  {'OK ' if richtig else 'XX '} {satz}")
            print(f"      expected: {erwartung}")
            if memories:
                for m in memories:
                    markierung = "  <-- name not the speaker!" if name not in m["text"] else ""
                    print(f"      got     : [{m['type']}] {m['text']}{markierung}")
            else:
                print("      got     : (nothing)")
    finally:
        mod.ASSISTANT_LANGUAGE_NAME = vorher

    return treffer, erwartet


def main():
    p = argparse.ArgumentParser(description="Probe the extractor in several languages")
    p.add_argument("--name", default=USER_NAME,
                   help=f"the speaker the facts are about (default: {USER_NAME})")
    p.add_argument("--lang", action="append",
                   help="limit to one language, repeatable (e.g. --lang English)")
    args = p.parse_args()

    if args.lang:
        sprachen, unbekannt = [], []
        for eingabe in args.lang:
            treffer = finde_sprache(eingabe)
            (sprachen if treffer else unbekannt).append(treffer or eingabe)
        if unbekannt:
            print(f"No probes for: {', '.join(unbekannt)}")
            print(f"Available: {', '.join(PROBEN)} "
                  f"(or the codes {', '.join(sorted(KUERZEL))})")
            raise SystemExit(1)
    else:
        sprachen = list(PROBEN)

    # A failed model call makes extract() return nothing, which here would
    # look exactly like "the model decided not to store anything". For a
    # script whose whole job is telling those two apart, that has to be
    # caught up front rather than puzzled over afterwards.
    if BACKEND == "openrouter" and not getattr(config, "OPENROUTER_API_KEY", ""):
        print("OPENROUTER_API_KEY is empty - every probe would come back\n"
              "empty and that would mean nothing. Put your key in config.py\n"
              "or set the environment variable, then run this again.")
        raise SystemExit(1)

    print(f"Backend: {BACKEND}")
    ergebnis = {}
    for sprache in sprachen:
        ergebnis[sprache] = pruefe(sprache, PROBEN[sprache], args.name)

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    for sprache, (treffer, gesamt) in ergebnis.items():
        print(f"  {sprache:10} {treffer}/{gesamt} probes behaved as expected")
    if len(ergebnis) > 1:
        werte = {t for t, _ in ergebnis.values()}
        if len(werte) > 1:
            print("\n  The languages differ. That is a finding - the probes mean\n"
                  "  the same thing in each, so the sentences are not the cause.")
        else:
            print("\n  No difference between the languages.")


if __name__ == "__main__":
    main()
