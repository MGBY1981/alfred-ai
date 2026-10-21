"""
language_packs.py

Language-dependent data for the memory heuristics.

WHY THIS FILE EXISTS
--------------------
Most of Alfred is language-agnostic: the prompts take their language from
ASSISTANT_LANGUAGE_NAME, and the memory filter only ever returns ID
numbers, never text, so it needs no translation at all.

Four pieces of logic are the exception. They do not ask a model, they
work on the text directly, and that means they need to know the language:

    non_subject_words   which word at the start of a sentence is NOT a
                        subject ("The user ..." has no name in it)
    verb_hints          which words mark a text as a real sentence, used
                        to spot subject-less fragments like "LEGO"
    time_pattern        which words mark a memory as time-bound, used by
                        alfred_sleep stage 3 (ageing)
    placeholder_rules   how to turn "the user" into the actual name

Before this file existed, all four were hard-coded German. With the
prompts set to English the German tables silently stopped matching -
stage 3 found nothing at all, and short English sentences were wrongly
flagged as fragments. Hence: one pack per language, picked by
ASSISTANT_LANGUAGE.

ADDING A LANGUAGE
-----------------
Copy the "en" pack, translate the four tables, and register it in PACKS
under its two-letter code. Nothing else needs touching.

If ASSISTANT_LANGUAGE names a language with no pack, get_pack() returns
NEUTRAL_PACK: the four heuristics above switch themselves OFF rather
than guessing with the wrong language's words. Everything that asks a
model keeps working, so the assistant still answers and still stores
memories - it just does without the local text tricks. Callers can see
this on the pack's `heuristics_available` flag and say so in their log.
"""

import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple


# ==========================================================
# German
# ==========================================================
# These tables are the original ones this project grew up with. The 43
# test cases of memory_extractor_regression_test_v5.py are German, and
# these heuristics are part of why they do well: German capitalizes
# every noun, so naive
# "look for a capitalized word" approaches produce constant false alarms
# (see the comment on _mehrere_subjekte in alfred_sleep.py).

_DE_NON_SUBJECT = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "eines", "dieser", "diese", "dieses", "diesem", "diesen", "er", "sie", "es",
    "wir", "ihr", "man", "alle", "beide", "jeder", "jede", "jedes", "kein",
    "keine", "mein", "meine", "sein", "seine", "unser", "unsere", "im", "in",
    "am", "an", "auf", "aus", "bei", "bis", "durch", "für", "gegen", "mit",
    "nach", "seit", "von", "vor", "zu", "zum", "zur", "über", "unter", "als",
    "wenn", "weil", "und", "aber", "auch", "nur", "noch", "schon", "ab",
    "heute", "morgen", "gestern", "damals", "früher", "inzwischen", "es",
    "hier", "dort", "da", "dann", "so", "wie", "was", "wer", "wo",
}

_DE_VERB_HINTS = {
    "mag", "mögen", "ist", "sind", "war", "waren", "wird", "werden", "hat",
    "haben", "hatte", "hatten", "macht", "machen", "spielt", "spielen",
    "spielte", "interessiert", "interessieren", "nutzt", "nutzen", "verwendet",
    "verwenden", "komponiert", "komponieren", "arbeitet", "arbeiten", "lebt",
    "leben", "wohnt", "wohnen", "trinkt", "trinken", "zeichnet", "zeichnen",
    "kann", "können", "liebt", "lieben", "sammelt", "sammeln", "fährt",
    "fahren", "geht", "gehen", "besitzt", "besitzen", "heißt", "heisst",
    "hört", "hören", "liest", "lesen", "schreibt", "schreiben", "programmiert",
    "programmieren", "baut", "bauen", "plant", "planen", "sucht", "suchen",
    "braucht", "brauchen", "möchte", "wollen", "will", "kocht", "kochen",
    "reist", "reisen", "läuft", "laufen", "malt", "malen", "gibt", "geben",
    "kümmert", "pflegt", "teilt", "teilen", "überwintert", "widmete",
}

_DE_TIME_PATTERN = re.compile(
    r"\b(januar|februar|märz|maerz|april|mai|juni|juli|august|september|oktober|november|dezember|"
    r"montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag|"
    r"morgen|übermorgen|nächste[nrs]?|kommende[nrs]?|heute|demnächst|bald|"
    r"20[0-9]{2})\b|\d{1,2}\.\d{1,2}\.",
    re.I,
)

# Nouns that appear as "die Frau des Nutzers" and should come out as
# "Jonas' Frau" rather than "Die Frau Jonas'".
_DE_RELATIVE_NOUNS = (
    "Frau|Ehefrau|Mann|Ehemann|Mutter|Vater|Eltern|Sohn|Tochter|Kinder|"
    "Schwester|Bruder|Familie|Fall|Wohnung|Haus|Auto|Arzt|Hausarzt"
)

_DE_PLACEHOLDER_GENITIVE = r"(?:Nutzers|Benutzers|Gesprächspartners|Users)"
_DE_PLACEHOLDER_NOMINATIVE = r"(?:Nutzer|Benutzer|Gesprächspartner|User)"

_DE_RELATIVE_PATTERN = re.compile(
    r"\b(die|der|das|Die|Der|Das) "
    r"(" + _DE_RELATIVE_NOUNS + r") "
    r"des " + _DE_PLACEHOLDER_GENITIVE + r"\b"
)


def _possessive_de(person: str) -> str:
    """German genitive of a personal name.

    Names ending in an s-sound take a bare apostrophe ("Jonas' Frau"),
    everything else takes an s ("Annas Mann"). Without this distinction
    the result reads "Jonass Frau".
    """
    return person + "'" if person[-1:].lower() in ("s", "ß", "x", "z") else person + "s"


def _relative_replacement_de(m: "re.Match", person: str) -> str:
    """Turns "die Frau des Nutzers" into "Jonas' Frau".

    The special case is the nested genitive chain: in "die Schwester der
    Frau des Nutzers" the simple replacement would produce "die Schwester
    Jonas' Frau" - grammatically wrong. The tell-tale sign is a noun
    standing immediately in front of it; since nouns are capitalized in
    German, a look at the preceding word is enough for that. In that case
    "von" is inserted.
    """
    davor = m.string[:m.start()].rstrip()
    woerter_davor = davor.split()
    letztes_wort = woerter_davor[-1] if woerter_davor else ""

    # Capitalized word directly in front and not the start of the sentence
    # -> a noun, hence a genitive chain.
    verkettet = bool(letztes_wort) and letztes_wort[:1].isupper() and len(woerter_davor) > 1

    if verkettet:
        return f"von {_possessive_de(person)} {m.group(2)}"
    return f"{_possessive_de(person)} {m.group(2)}"


def _placeholder_rules_de(person: str) -> List[Tuple]:
    """Replacements for placeholder subjects.

    The extractor frequently writes "Der Nutzer ..." instead of the name.
    Within one user_id partition that can be resolved unambiguously, which
    makes it one of the few text changes one can take responsibility for
    without a model. The order matters: relatives before genitive before
    dative before nominative, otherwise the shorter rule applies first.
    """
    return [
        (_DE_RELATIVE_PATTERN, lambda m: _relative_replacement_de(m, person)),
        (re.compile(r"\bdes " + _DE_PLACEHOLDER_GENITIVE + r"\b", re.I),
         _possessive_de(person)),
        (re.compile(r"\b(?:dem|den) " + _DE_PLACEHOLDER_NOMINATIVE + r"\b", re.I),
         person),
        (re.compile(r"\b(?:der|Der) " + _DE_PLACEHOLDER_NOMINATIVE + r"\b"),
         person),
    ]


# ==========================================================
# English
# ==========================================================

_EN_NON_SUBJECT = {
    "the", "a", "an", "this", "that", "these", "those", "he", "she", "it",
    "they", "we", "you", "i", "his", "her", "its", "their", "our", "my",
    "your", "all", "both", "each", "every", "no", "none", "someone",
    "somebody", "anyone", "anybody", "everyone", "everybody", "one",
    "in", "on", "at", "to", "from", "by", "with", "without", "for",
    "against", "about", "over", "under", "after", "before", "since",
    "during", "until", "between", "through", "as", "if", "when", "while",
    "because", "and", "but", "also", "only", "still", "already", "then",
    "there", "here", "now", "today", "tomorrow", "yesterday", "earlier",
    "later", "so", "how", "what", "who", "where", "why",
}

# Present and past tense of the verbs that actually turn up in stored
# facts, plus the auxiliaries. Third person singular matters most, since
# facts are written as "<who> <does what>".
_EN_VERB_HINTS = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "does", "do", "did", "will", "would", "can", "could", "may", "might",
    "should", "must",
    "likes", "like", "liked", "loves", "love", "loved", "prefers", "prefer",
    "enjoys", "enjoy", "enjoyed", "hates", "hate", "wants", "want", "wanted",
    "needs", "need", "needed", "owns", "own", "owned", "uses", "use", "used",
    "works", "work", "worked", "lives", "live", "lived", "stays", "stay",
    "drinks", "drink", "drank", "eats", "eat", "ate", "cooks", "cook",
    "plays", "play", "played", "reads", "read", "writes", "write", "wrote",
    "draws", "draw", "drew", "paints", "paint", "painted", "builds", "build",
    "built", "makes", "make", "made", "plans", "plan", "planned", "runs",
    "run", "ran", "walks", "walk", "drives", "drive", "drove", "rides",
    "ride", "flies", "fly", "flew", "travels", "travel", "travelled",
    "traveled", "visits", "visit", "visited", "moves", "move", "moved",
    "studies", "study", "studied", "teaches", "teach", "taught", "learns",
    "learn", "learned", "learnt", "speaks", "speak", "spoke", "listens",
    "listen", "watches", "watch", "collects", "collect", "collected",
    "composes", "compose", "programs", "program", "codes", "code",
    "develops", "develop", "manages", "manage", "keeps", "keep", "kept",
    "takes", "take", "took", "gives", "give", "gave", "gets", "get", "got",
    "goes", "go", "went", "comes", "come", "came", "knows", "know", "knew",
    "thinks", "think", "believes", "believe", "cares", "care", "shares",
    "share", "supports", "support", "follows", "follow", "started", "starts",
    "finished", "finishes", "called", "calls", "named", "interested",
    "interests", "lies", "sits", "stands",
}

_EN_TIME_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"tomorrow|today|yesterday|tonight|"
    r"next|coming|upcoming|soon|shortly|recently|"
    r"20[0-9]{2})\b"
    r"|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.I,
)

_EN_RELATIVE_NOUNS = (
    "wife|husband|spouse|mother|father|parents|son|daughter|children|kids|"
    "sister|brother|family|flat|apartment|house|home|car|doctor|dentist"
)

_EN_RELATIVE_PATTERN = re.compile(
    r"\bthe (" + _EN_RELATIVE_NOUNS + r") of the (?:user|conversation partner)\b",
    re.I,
)


def _possessive_en(person: str) -> str:
    """English possessive. "Jonas's wife" follows modern usage; a name
    already ending in s keeps the extra s rather than dropping it."""
    return person + "'s"


def _placeholder_rules_en(person: str) -> List[Tuple]:
    """English counterpart of _placeholder_rules_de.

    English needs no case handling, so this is far shorter. The order
    still matters: the "of the user" form first, then the possessive,
    then the bare subject.
    """
    possessive = _possessive_en(person)
    return [
        (_EN_RELATIVE_PATTERN, lambda m: f"{possessive} {m.group(1)}"),
        (re.compile(r"\bthe (?:user|conversation partner)'s\b", re.I), possessive),
        (re.compile(r"\bthe (?:user|conversation partner)\b", re.I), person),
    ]


# ==========================================================
# Neutral fallback
# ==========================================================
# Used when ASSISTANT_LANGUAGE names a language nobody has written a pack
# for. Empty tables plus heuristics_available=False, so callers can turn
# the affected checks off instead of running them with the wrong
# language's vocabulary - which is worse than not running them at all.

NEUTRAL_PACK: Dict[str, object] = {
    "code": "",
    "name": "unknown",
    "heuristics_available": False,
    "non_subject_words": set(),
    "verb_hints": set(),
    "time_pattern": None,
    "placeholder_rules": lambda person: [],
}


PACKS: Dict[str, Dict[str, object]] = {
    "de": {
        "code": "de",
        "name": "German",
        "heuristics_available": True,
        "non_subject_words": _DE_NON_SUBJECT,
        "verb_hints": _DE_VERB_HINTS,
        "time_pattern": _DE_TIME_PATTERN,
        "placeholder_rules": _placeholder_rules_de,
    },
    "en": {
        "code": "en",
        "name": "English",
        "heuristics_available": True,
        "non_subject_words": _EN_NON_SUBJECT,
        "verb_hints": _EN_VERB_HINTS,
        "time_pattern": _EN_TIME_PATTERN,
        "placeholder_rules": _placeholder_rules_en,
    },
}


def get_pack(code: Optional[str]) -> Dict[str, object]:
    """The pack for a language code, or NEUTRAL_PACK if there is none.

    Only the first two letters are looked at, so "en-GB" and "de_DE"
    find their pack as well.
    """
    schluessel = (code or "").strip().lower().replace("_", "-")[:2]
    return PACKS.get(schluessel, NEUTRAL_PACK)


def available_languages() -> List[str]:
    """The language codes that have a pack, for log output and the UI."""
    return sorted(PACKS)
