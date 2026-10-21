"""
alfred_sleep.py

Alfred 2.0 - memory consolidation ("sleep")

Analogy: one reason humans sleep is to consolidate memories - what belongs
together is merged, the unimportant fades, and the rest is filed away more
cleanly. That is exactly what this module does to the Chroma database.

BASIC PRINCIPLE: PLANNING AND APPLYING ARE SEPARATE
---------------------------------------------------
This module NEVER changes the database as a side effect. It first produces
a plan (a pure read operation whose result is a JSON file) which can be
reviewed, corrected and selected or deselected item by item. Only a
separate call to apply_plan() writes.

The reason is not caution for its own sake: one consolidation run makes
dozens of individual decisions per pass, and every single one of them can
destroy a real fact. A plan that can be proof-read is the only way to be
able to judge that at all.

THREE STAGES
------------
Stage 1 "mechanical"   - no LLM. Deterministic rules that cannot invent
                         anything: normalized duplicates and placeholder
                         subjects ("Der Nutzer ..." -> "Jonas ...").
                         This stage is unit-testable and practically
                         risk-free.

Stage 2 "merge"        - LLM. Two passes: first find groups (cheap model,
                         returns nothing but ID lists), then formulate one
                         consolidated fact per group (good model). The
                         split into two is deliberately the same split as
                         in the filter and the extractor - the grouping
                         step is easy to proof-read on its ID output.

Stage 3 "age"          - LLM. Time-bound facts that have expired ("The
                         trip goes to Crete on 4 August") are converted
                         into the past tense or - in the case of pure
                         logistics without lasting value - deleted. Sleep
                         does not only consolidate, it also forgets.

WHAT DELIBERATELY DOES NOT HAPPEN
---------------------------------
- No detour via memory_export.py / memory_import.py. A round trip through
  memory_import.py currently sets every created_at to the time of import
  (the original value ends up in "original_created_at", which no other
  file in the project reads). That would destroy Alfred's only mechanism
  for resolving conflicts - "the newer statement is the more current one".
- No delete-then-rewrite. apply_plan() ALWAYS creates the new entry first
  and deletes the sources afterwards. If something aborts halfway through,
  duplicates are left behind - annoying, but repairable. The other way
  round, the fact would be gone.
- No embedding similarity for forming the groups. The vectors in this DB
  are a mixture of real ones (written via alfred_ollama.py, i.e.
  mem0/Ollama) and zero vectors (written via alfred.py) and worthless
  as a similarity measure.

USAGE
-----
    python alfred_sleep.py                      Produce a plan (read-only)
    python alfred_sleep.py --stages 1           Only the mechanical stage
    python alfred_sleep.py --apply plan.json    Apply a plan
    python alfred_sleep.py --report plan.json   View a plan as plain text

Or through the user interface:

    streamlit run alfred_sleep_ui.py

IMPORTANT: the Alfred chat should be closed while a plan is being applied.
Two processes writing into the same Chroma/SQLite file at the same time are
an avoidable source of errors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import chromadb

from config_guard import require_config
require_config()   # readable message if config.py is missing

import config
from config import CHROMA_DB_DIR, USER_ID, USER_NAME

# Read with getattr so an older config.py without this setting keeps
# working - it then uses the name the project has always used.
CHROMA_COLLECTION = getattr(config, "CHROMA_COLLECTION", "alfred_memory")
from language_packs import get_pack
from memory_extractor_v5 import ALLOWED_TYPES
from openrouter_client import OpenRouterError, chat_completion, extract_text, extract_usage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAN_DIR = os.path.join(BASE_DIR, "sleep_plans")
BACKUP_DIR = os.path.join(BASE_DIR, "sleep_backups")


# ==========================================================
# Configuration
#
# Deliberately read out of config.py with getattr instead of extending
# config.py: that way the backwards-compatibility promise made there stays
# untouched and this module also runs against an older config.py. Anyone
# who wants to change the values permanently simply enters them in
# config.py - they then automatically take precedence over the defaults
# here.
# ==========================================================

def _cfg(name: str, default: Any) -> Any:
    return getattr(config, name, default)


# The language Alfred answers and stores in. Both are read with _cfg so
# that this module still runs against a config.py predating the language
# setting - it then simply works in English.
ASSISTANT_LANGUAGE = _cfg("ASSISTANT_LANGUAGE", "en")
ASSISTANT_LANGUAGE_NAME = _cfg("ASSISTANT_LANGUAGE_NAME", "English")

# The language-dependent word lists for the three local heuristics
# (subject detection, fragment detection, time reference) and for the
# placeholder rewriting in stage 1. See language_packs.py for what
# happens with a language that has no pack.
SPRACHPAKET = get_pack(ASSISTANT_LANGUAGE)
HEURISTIKEN_VERFUEGBAR = bool(SPRACHPAKET["heuristics_available"])


# Finds groups of memories that belong together. That is a pure
# recognition task with numbers as its output - a cheap model is enough
# here, and it carries the largest token load (the complete list of
# memories). A comparison test on real data clearly spoke in favour of
# Flash here - see the table of measurements in config.example.py.
SLEEP_GROUP_MODEL = _cfg("SLEEP_GROUP_MODEL", "google/gemini-2.5-flash")

# Formulates the consolidated fact out of a group. This is the hardest
# spot in the whole project: a judgement task (is this detail redundant or
# does it stand on its own?) plus generation (write a sentence that loses
# nothing), and the error is SILENT - a detail that has fallen away leaves
# no trace. Unlike in the chat, this runs at most once a month, and one run
# costs cents even with the most expensive model. That is why the big model
# is used here on purpose, while the chat stays cheap.
SLEEP_MERGE_MODEL = _cfg("SLEEP_MERGE_MODEL", "anthropic/claude-opus-5")

# Judges whether a time-bound fact has expired. Likewise a judgement task,
# with deletions without replacement as a possible consequence.
SLEEP_AGE_MODEL = _cfg("SLEEP_AGE_MODEL", "anthropic/claude-opus-5")

# Selection lists for the user interface. Fixed versions instead of
# "~...-latest", so that a model does not shift under you between two
# comparison runs - for an operation that runs once a month,
# reproducibility is worth more than being up to date. This is the place
# to enter new models; the user interface reads these lists.
SLEEP_MODELLE_GRUPPIERUNG = _cfg("SLEEP_MODELLE_GRUPPIERUNG", [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-5",
])

SLEEP_MODELLE_URTEIL = _cfg("SLEEP_MODELLE_URTEIL", [
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-haiku-4.5",
    "~anthropic/claude-haiku-latest",
    "google/gemini-2.5-flash",
])

SLEEP_TIMEOUT = _cfg("SLEEP_TIMEOUT", 120)

# How many memories per grouping call. With the current not quite 400
# facts everything fits into a single call; the value is the ripcord for
# later. Before splitting, the list is sorted alphabetically so that
# similar texts end up in the same block where possible.
SLEEP_GROUP_BATCH = _cfg("SLEEP_GROUP_BATCH", 250)

# How many groups per merge call.
SLEEP_MERGE_BATCH = _cfg("SLEEP_MERGE_BATCH", 10)

# From which model confidence onwards an LLM action is pre-ticked in the
# plan. Below that it has to be ticked deliberately.
SLEEP_AUTO_ACCEPT_CONFIDENCE = _cfg("SLEEP_AUTO_ACCEPT_CONFIDENCE", 0.8)

# How many actions are written in one go while applying. Larger = faster,
# but coarser progress and, in case of an error, a larger affected block.
# 25 is a usable compromise.
SLEEP_APPLY_BLOCK = _cfg("SLEEP_APPLY_BLOCK", 25)


# ==========================================================
# Database access (identical pattern to memory_viewer.py)
# ==========================================================

def get_collection():
    collection_name = CHROMA_COLLECTION
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(collection_name)


def load_memories(collection, user_id: str) -> List[Dict[str, Any]]:
    """All memories of one user, oldest first.

    Sorted ascending, because when consolidating, the order in which
    things came into being is the interesting reading direction - in the
    chat it is the other way round.
    """
    raw = collection.get(where={"user_id": user_id}, include=["metadatas", "documents"])

    ids = raw.get("ids") or []
    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents") or []

    items = []
    for mem_id, md, doc in zip(ids, metadatas, documents):
        md = md or {}
        text = md.get("data") or doc or ""
        if not text:
            continue
        items.append({
            "id": mem_id,
            "text": text,
            "type": md.get("type", "other"),
            "created_at": md.get("created_at"),
            "edited_at": md.get("edited_at"),
        })

    items.sort(key=lambda m: m.get("created_at") or "")
    return items


def get_embedding_dimension(collection) -> int:
    """Read the vector dimension off an existing entry instead of guessing.

    Taken over from alfred.py: the content of the vector is
    meaningless for the purposes here (similarity is never searched for),
    but the dimension has to match the collection, otherwise Chroma
    aborts.
    """
    probe = collection.get(limit=1, include=["embeddings"])
    embeddings = probe.get("embeddings")
    if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
        return len(embeddings[0])
    return 768


def normalize_for_dedup(text: str) -> str:
    """Comparison form for the duplicate check - identical to the one in
    alfred.py / alfred_ollama.py / memory_import.py."""
    t = unicodedata.normalize("NFKC", text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t.rstrip(".!?;:, ")


def format_datum(raw: Optional[str]) -> str:
    if not raw:
        return "?"
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).strftime("%d.%m.%Y")
    except Exception:
        return str(raw)


def neuestes_datum(quellen: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The merged entry inherits the MOST RECENT created_at of its sources.

    Reasoning: Alfred's system prompt interprets the date as "where
    statements contradict each other, the newer one is the more current
    one". A merged fact reflects the most current state of knowledge and
    must therefore also be dated that way. The oldest date of the group is
    preserved in the metadata (see apply_plan).
    """
    daten = [q.get("created_at") for q in quellen if q.get("created_at")]
    return max(daten) if daten else None


# ==========================================================
# Stage 1: mechanical (no LLM)
# ==========================================================

def _platzhalter_regeln(person: str) -> List[tuple]:
    """Replacements for placeholder subjects, taken from the language pack.

    The extractor frequently writes "the user ..." instead of the name.
    Within one user_id partition that can be resolved unambiguously, which
    makes it one of the few text changes one can take responsibility for
    without a model.

    The rules themselves live in language_packs.py, because how a
    placeholder is phrased - and what has to happen to the rest of the
    sentence when it is replaced - is a question of grammar. A language
    without a pack gets an empty list, so stage 1 then only looks for
    duplicates and leaves placeholders alone.
    """
    return SPRACHPAKET["placeholder_rules"](person)


def plan_mechanisch(memories: List[Dict[str, Any]], person: str) -> List[Dict[str, Any]]:
    """Stage 1. Finds normalized duplicates and placeholder subjects.

    Invents nothing and needs no model - the result is always identical
    for the same input.
    """
    aktionen: List[Dict[str, Any]] = []
    verbraucht: set = set()

    # --- 1a. Normalized duplicates ---------------------------------------
    nach_norm: Dict[str, List[Dict[str, Any]]] = {}
    for m in memories:
        nach_norm.setdefault(normalize_for_dedup(m["text"]), []).append(m)

    for _, gruppe in nach_norm.items():
        if len(gruppe) < 2:
            continue

        # Keep the most recent entry, pull in the older ones.
        gruppe = sorted(gruppe, key=lambda m: m.get("created_at") or "")
        behalten = gruppe[-1]

        aktionen.append({
            "stufe": "mechanical",
            "aktion": "merge",
            "quellen": gruppe,
            "neu": {"text": behalten["text"], "type": behalten["type"]},
            "created_at": neuestes_datum(gruppe),
            "confidence": 1.0,
            "begruendung": (
                f"{len(gruppe)} identical entries (after normalizing "
                f"capitalization and punctuation)."
            ),
            "pruefen": False,
        })
        verbraucht.update(m["id"] for m in gruppe)

    # --- 1b. Placeholder subjects ----------------------------------------
    regeln = _platzhalter_regeln(person)

    for m in memories:
        if m["id"] in verbraucht:
            continue

        neu_text = m["text"]
        for muster, ersatz in regeln:
            neu_text = muster.sub(ersatz, neu_text)

        if neu_text == m["text"]:
            continue

        aktionen.append({
            "stufe": "mechanical",
            "aktion": "rewrite",
            "quellen": [m],
            "neu": {"text": neu_text, "type": m["type"]},
            "created_at": m.get("created_at"),
            "confidence": 1.0,
            "begruendung": (
                f"Placeholder subject replaced with '{person}' - unambiguous "
                f"within this user_id."
            ),
            "pruefen": False,
        })
        verbraucht.add(m["id"])

    return aktionen


# ==========================================================
# LLM helpers
# ==========================================================

def _json_aus_antwort(text: str) -> Dict[str, Any]:
    """Robust JSON extraction - same logic as in memory_manager_v2_1.py."""
    cleaned = (text or "").strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    ende = cleaned.rfind("}")
    if start == -1 or ende == -1:
        raise ValueError("No JSON object found in the response.")

    return json.loads(cleaned[start:ende + 1])


def _frage_modell(model: str, system_prompt: str, user_prompt: str,
                  usage_sammler: Optional[List] = None) -> Dict[str, Any]:
    data = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        timeout=SLEEP_TIMEOUT,
        retries=2,
    )
    if usage_sammler is not None:
        usage_sammler.append(extract_usage(data))
    return _json_aus_antwort(extract_text(data))


# ==========================================================
# Stage 2: grouping and merging
# ==========================================================

GRUPPEN_SYSTEM_PROMPT = """You are Alfred's sleep centre. Your ONLY task is to find, in a numbered list of stored memories, those groups which concern the same matter and could sensibly be combined into ONE memory.

RULES

- Never answer questions about content. Never rephrase any text. Return numbers only.
- MOST IMPORTANT RULE: NEVER group facts about DIFFERENT PEOPLE. "Jonas likes football" and "Lena likes football" are TWO facts and must never be put in the same group. If you are not sure whether two facts concern the same person, form NO group.
- Only group what really concerns the same matter: literal duplicates, rephrasings of the same fact, fragments of an originally connected enumeration (e.g. "Italy", "Croatia", "Adriatic" after a single statement about travel destinations), or a later statement that supersedes an earlier one.
- Being related in topic is NO reason. "Jonas is a Java developer" and "Jonas is building an AI assistant" both belong to the topic of software, but they are two independent facts.
- A group has at least 2 entries. Everything that stands on its own belongs in NO group - that is the normal case, not the exception.
- Each number may appear in at most ONE group.
- When in doubt, no group. A missed duplicate costs a little space; a wrong group costs a real fact.
- Write the result exclusively in {language}.

OUTPUT

Exclusively JSON in this format:

{"gruppen":[{"ids":[3,17,42],"thema":"Travel destinations around the Mediterranean"},{"ids":[8,9],"thema":"Employer"}]}

If you find no group: {"gruppen":[]}""".replace("{language}", ASSISTANT_LANGUAGE_NAME)


VERSCHMELZ_SYSTEM_PROMPT = """You are Alfred's sleep centre. You are given groups of stored memories which concern the same matter. Formulate ONE consolidated memory per group.

RULES

- Only use information that is contained in the source texts of the group. Invent nothing, add nothing, bring in no world knowledge, infer nothing.
- NO INFORMATION MAY BE LOST. If the group cannot be summarized without a detail disappearing, set "abbrechen": true - then everything stays unchanged. That is a perfectly acceptable outcome.
- The result MUST be a complete sentence with an explicitly named subject and a verb. A text like "LEGO" or "Champions League" is INVALID.
- Read your result on its own, without knowing the source texts: is it clear WHO it is about and WHAT it refers to? If not, add that from the source texts - but do not invent anything on top.
- If the source texts CONTRADICT each other, the statement with the most recent date applies. Do not mention the superseded state.
- If the source texts concern different people, set "abbrechen": true.
- The group's topic label is NOT a source. If there is no agent in the source texts themselves (e.g. with bare keywords such as "Italy", "LEGO", "Champions League"), you may not take a subject from the topic label and may not guess one either. In that case set "pruefen": true and point out explicitly in the reasoning where the subject came from. If the keywords do not obviously belong together in substance, set "abbrechen": true instead - bundling travel destinations, sports and toys into one sentence does not produce a usable fact.
- "type" MUST be exactly one of these values: profession, workplace, family, device, interest, hobby, project, goal, skill, preference, other. Never invent your own types. Choose the type that fits the consolidated text - it may differ from the types of the sources.
- Set "pruefen": true if you were unsure about anything - in particular if you had to add a subject or a reference that was not contained literally in the source texts.
- Write the result exclusively in {language}.

OUTPUT

Exclusively JSON in this format:

{"ergebnisse":[{"gruppe":1,"text":"Jonas likes travelling to the Mediterranean, among other places to Italy, Croatia and Greece.","type":"interest","confidence":0.9,"begruendung":"Six fragments of an enumeration combined into one sentence.","pruefen":false,"abbrechen":false}]}""".replace("{language}", ASSISTANT_LANGUAGE_NAME)


# Words that can stand at the beginning of a sentence without being a
# subject - from the language pack, see language_packs.py.
_KEIN_SUBJEKT = SPRACHPAKET["non_subject_words"]


def _subjekt(text: str) -> Optional[str]:
    """The first word of a fact, provided it is fit to be a subject.

    The extractor almost always writes facts in the form "<who> <does
    what>", so the first word is as a rule the agent. That is good enough
    as an approximation - this is not about grammar analysis, only about
    the question "are these source texts talking about the same person?".

    The apostrophe is deliberately KEPT, which is the whole point of the
    check: "Anna" and "Anna's" have to come out as two different
    subjects, because one is Anna and the other is a relative of hers.
    Stripping it made both read "Anna", and a group mixing the two - the
    most dangerous kind of merge there is - sailed through unflagged.
    Keeping it also means names like O'Brien are recognised at all.
    """
    woerter = (text or "").strip().split()
    if not woerter:
        return None

    wort = woerter[0].strip(".,!?;:„\"()")
    kern = wort.replace("-", "").replace("'", "").replace("’", "")
    if not wort or not wort[0].isupper() or not kern.isalpha():
        return None
    if wort.lower() in _KEIN_SUBJEKT:
        return None
    return wort


def _mehrere_subjekte(texte: Sequence[str]) -> bool:
    """True if the source texts are talking about different agents.

    This is the safety net against the most dangerous class of error:
    merging facts about different family members.

    The PREVIOUS version simply looked for capitalized words - in German,
    that is to say for nouns. That made "Kumquat", "Java" and "Champions
    League" count as personal names, and practically every group was
    flagged as suspicious. A warning that goes off almost every time is no
    longer a warning but noise. That is why only the subject is used now.
    """
    # "Jonas ..." and "Jonas' Frau ..." deliberately stay TWO subjects -
    # one is Jonas, the other his wife. A group that mixes the two
    # deserves a look.
    subjekte = {s for s in (_subjekt(t) for t in texte) if s}
    return len(subjekte) > 1


# Verbs whose presence suggests that a text is a complete sentence - from
# the language pack, the same table memory_viewer.py uses.
_VERB_HINWEISE = SPRACHPAKET["verb_hints"]


def _ist_fragment(text: str) -> bool:
    """Rough detection of fragments without a subject or verb such as
    "LEGO" or "Champions League" - the same class of error that
    memory_viewer.py already tracks down.

    A group containing such fragments ALWAYS needs a look: to turn them
    into a complete sentence a subject is required, and that subject
    cannot come from the source texts - it would be invented.

    Without a verb list for the configured language the check reports
    nothing instead of everything: with an empty table every short text
    would come out as a fragment, and a warning that fires on every group
    is worse than no warning.
    """
    if not _VERB_HINWEISE:
        return False
    woerter = (text or "").strip().rstrip(".!?").split()
    if not woerter or len(woerter) > 3:
        return False
    return not any(w.lower().strip(".,!?") in _VERB_HINWEISE for w in woerter)


def plan_verschmelzen(memories: List[Dict[str, Any]],
                      ausgenommen: Optional[set] = None,
                      group_model: str = None,
                      merge_model: str = None,
                      usage_sammler: Optional[List] = None,
                      fortschritt=None) -> List[Dict[str, Any]]:
    """Stage 2. First find groups, then consolidate each group."""

    ausgenommen = ausgenommen or set()
    group_model = group_model or SLEEP_GROUP_MODEL
    merge_model = merge_model or SLEEP_MERGE_MODEL

    kandidaten = [m for m in memories if m["id"] not in ausgenommen]
    if len(kandidaten) < 2:
        return []

    # Sort alphabetically so that similar texts end up in the same block
    # where possible - only relevant once there are more than
    # SLEEP_GROUP_BATCH facts and the list actually has to be split.
    kandidaten = sorted(kandidaten, key=lambda m: normalize_for_dedup(m["text"]))

    gruppen: List[Dict[str, Any]] = []
    benutzt: set = set()

    bloecke = [kandidaten[i:i + SLEEP_GROUP_BATCH]
               for i in range(0, len(kandidaten), SLEEP_GROUP_BATCH)]

    for nr, block in enumerate(bloecke, 1):
        if fortschritt:
            fortschritt(f"Searching for groups ({nr}/{len(bloecke)}, {len(block)} memories)")

        liste = "\n".join(
            f"{i}: [{m['type']}] {format_datum(m['created_at'])}: {m['text']}"
            for i, m in enumerate(block)
        )

        try:
            antwort = _frage_modell(
                group_model,
                GRUPPEN_SYSTEM_PROMPT,
                f"Stored memories (numbered):\n{liste}",
                usage_sammler,
            )
        except (OpenRouterError, ValueError, json.JSONDecodeError):
            # A block that fails must not abort the run.
            continue

        for g in antwort.get("gruppen", []):
            if not isinstance(g, dict):
                continue

            mitglieder = []
            for roh in g.get("ids", []):
                try:
                    idx = int(roh)
                except (TypeError, ValueError):
                    continue
                if idx < 0 or idx >= len(block):
                    continue
                m = block[idx]
                if m["id"] in benutzt:
                    continue
                benutzt.add(m["id"])
                mitglieder.append(m)

            if len(mitglieder) >= 2:
                gruppen.append({"thema": g.get("thema", ""), "mitglieder": mitglieder})

    if not gruppen:
        return []

    # --- Second pass: merging --------------------------------------------
    aktionen: List[Dict[str, Any]] = []

    pakete = [gruppen[i:i + SLEEP_MERGE_BATCH]
              for i in range(0, len(gruppen), SLEEP_MERGE_BATCH)]

    for nr, paket in enumerate(pakete, 1):
        if fortschritt:
            fortschritt(f"Summarizing groups ({nr}/{len(pakete)})")

        teile = []
        for i, g in enumerate(paket, 1):
            zeilen = "\n".join(
                f"  - [{m['type']}] {format_datum(m['created_at'])}: {m['text']}"
                for m in g["mitglieder"]
            )
            teile.append(f"Group {i} (topic according to the pre-selection: {g['thema']}):\n{zeilen}")

        try:
            antwort = _frage_modell(
                merge_model,
                VERSCHMELZ_SYSTEM_PROMPT,
                "\n\n".join(teile),
                usage_sammler,
            )
        except (OpenRouterError, ValueError, json.JSONDecodeError):
            continue

        for e in antwort.get("ergebnisse", []):
            if not isinstance(e, dict) or e.get("abbrechen"):
                continue

            try:
                gruppe = paket[int(e.get("gruppe", 0)) - 1]
            except (TypeError, ValueError, IndexError):
                continue

            text = (e.get("text") or "").strip()
            typ = e.get("type", "other")

            if not text or typ not in ALLOWED_TYPES:
                continue

            quelltexte = [m["text"] for m in gruppe["mitglieder"]]

            # A result that is noticeably shorter than the longest source
            # text has probably lost something. When summarizing
            # near-duplicates, on the other hand, a result of equal length
            # or minimally shorter is the normal case - which is why the
            # warning only kicks in from a perceptible gap onwards and not
            # at every single character.
            laengste_quelle = max(len(t) for t in quelltexte)
            verdacht_kuerzung = len(text) < 0.8 * laengste_quelle

            mehrere_subjekte = _mehrere_subjekte(quelltexte)
            hat_fragmente = any(_ist_fragment(t) for t in quelltexte)

            pruefen = bool(
                e.get("pruefen")
                or verdacht_kuerzung
                or mehrere_subjekte
                or hat_fragmente
                or float(e.get("confidence", 1.0)) < SLEEP_AUTO_ACCEPT_CONFIDENCE
            )

            hinweise = []
            if verdacht_kuerzung:
                hinweise.append(
                    f"result {len(text)} characters, longest source {laengste_quelle}"
                )
            if mehrere_subjekte:
                hinweise.append("source texts talk about different people")
            if hat_fragmente:
                hinweise.append(
                    "group contains fragments without a subject - the subject in "
                    "the result cannot come from the sources"
                )

            begruendung = e.get("begruendung", "")
            if hinweise:
                begruendung += " [Automatic note: " + "; ".join(hinweise) + "]"

            aktionen.append({
                "stufe": "merge",
                "aktion": "merge",
                "quellen": gruppe["mitglieder"],
                "neu": {"text": text, "type": typ},
                "created_at": neuestes_datum(gruppe["mitglieder"]),
                "confidence": float(e.get("confidence", 1.0)),
                "begruendung": begruendung.strip(),
                "pruefen": pruefen,
            })

    return aktionen


# ==========================================================
# Stage 3: ageing
# ==========================================================

ALTERN_SYSTEM_PROMPT = """You are Alfred's sleep centre. You are given stored memories which contain a reference to time, each with the date on which it was stored. Judge for each of them whether it has been superseded by the passage of time.

All memories are kept permanently in a database and are later presented to the assistant as facts that are CURRENTLY VALID. A sentence like "The trip goes to Crete on 4 August" is therefore misleading a year later - it sounds like an upcoming trip.

POSSIBLE ACTIONS

- "rewrite": the fact has lasting value but is phrased in the wrong tense or without a year. Rephrase it in the past tense and add the year if it follows unambiguously from the storage date. Example: "The trip goes to Crete on 4 August." (stored 29.07.2026) becomes "Jonas was on Crete in early August 2026."
- "delete": pure logistical details without lasting value which nobody is interested in any more after the event - flight numbers, times of individual appointments, departure airports, booking details.
- "behalten": the fact is timelessly valid or still applies. When in doubt, always this one.

RULES

- Invent nothing. You may only add a year if it follows inevitably from the storage date.
- Recurring things have NOT expired ("The tricky time is usually in January", "Jonas has his birthday in May").
- A past event is not automatically worthless. That someone was in Macedonia in 2023 remains a valid fact - at most it needs "rewrite", never "delete".
- Never "delete" personal or health-related statements, even if they contain a date. At most "rewrite".
- When in doubt, "behalten".
- Write the result exclusively in {language}.

OUTPUT

Exclusively JSON:

{"ergebnisse":[{"nr":3,"aktion":"rewrite","text":"Jonas was on Crete with his family in early August 2026.","type":"other","confidence":0.9,"begruendung":"The trip lies in the past.","pruefen":false}]}

You may leave out entries with "behalten".""".replace("{language}", ASSISTANT_LANGUAGE_NAME)


# Rough pre-filter: only memories with a recognizable reference to time go
# to the model. Saves tokens and keeps the run manageable. The pattern
# comes from the language pack - month and weekday names are the most
# language-bound thing in the whole project.
_ZEITBEZUG = SPRACHPAKET["time_pattern"]


def zeitbezogene(memories: List[Dict[str, Any]],
                 ausgenommen: Optional[set] = None) -> List[Dict[str, Any]]:
    """The memories that carry a reference to time, the input of stage 3.

    Returns an empty list if the configured language has no pack: stage 3
    then finds nothing and does nothing, which is the honest outcome. The
    alternative - searching English text for "nächste Woche" - looks like
    it works and quietly does nothing at all.
    """
    if _ZEITBEZUG is None:
        return []
    ausgenommen = ausgenommen or set()
    return [m for m in memories
            if m["id"] not in ausgenommen and _ZEITBEZUG.search(m["text"] or "")]


def plan_altern(memories: List[Dict[str, Any]],
                ausgenommen: Optional[set] = None,
                model: str = None,
                heute: Optional[datetime] = None,
                usage_sammler: Optional[List] = None,
                fortschritt=None) -> List[Dict[str, Any]]:
    """Stage 3. Rephrase or remove expired time-bound facts."""

    model = model or SLEEP_AGE_MODEL
    heute = heute or datetime.now(timezone.utc)

    kandidaten = zeitbezogene(memories, ausgenommen)
    if not kandidaten:
        return []

    if fortschritt:
        fortschritt(f"Checking time-bound facts ({len(kandidaten)})")

    liste = "\n".join(
        f"{i}: [{m['type']}] stored on {format_datum(m['created_at'])}: {m['text']}"
        for i, m in enumerate(kandidaten)
    )

    try:
        antwort = _frage_modell(
            model,
            ALTERN_SYSTEM_PROMPT,
            f"Today is {heute.strftime('%d.%m.%Y')}.\n\n"
            f"Time-bound memories:\n{liste}",
            usage_sammler,
        )
    except (OpenRouterError, ValueError, json.JSONDecodeError):
        return []

    aktionen: List[Dict[str, Any]] = []
    gesehen: set = set()

    for e in antwort.get("ergebnisse", []):
        if not isinstance(e, dict):
            continue

        try:
            idx = int(e.get("nr", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(kandidaten) or idx in gesehen:
            continue

        art = e.get("aktion")
        if art not in ("rewrite", "delete"):
            continue

        gesehen.add(idx)
        quelle = kandidaten[idx]

        if art == "delete":
            aktionen.append({
                "stufe": "age",
                "aktion": "delete",
                "quellen": [quelle],
                "neu": None,
                "created_at": quelle.get("created_at"),
                "confidence": float(e.get("confidence", 1.0)),
                "begruendung": e.get("begruendung", ""),
                # Deleting without replacement is always a deliberate
                # decision.
                "pruefen": True,
            })
            continue

        text = (e.get("text") or "").strip()
        typ = e.get("type", quelle["type"])
        if not text or typ not in ALLOWED_TYPES:
            continue

        aktionen.append({
            "stufe": "age",
            "aktion": "rewrite",
            "quellen": [quelle],
            "neu": {"text": text, "type": typ},
            "created_at": quelle.get("created_at"),
            "confidence": float(e.get("confidence", 1.0)),
            "begruendung": e.get("begruendung", ""),
            "pruefen": bool(e.get("pruefen")) or
                       float(e.get("confidence", 1.0)) < SLEEP_AUTO_ACCEPT_CONFIDENCE,
        })

    return aktionen


# ==========================================================
# Assembling, saving and loading a plan
# ==========================================================

def erzeuge_plan(collection, user_id: str, person: Optional[str] = None,
                 stufen: Sequence[int] = (1, 2, 3),
                 group_model: Optional[str] = None,
                 merge_model: Optional[str] = None,
                 age_model: Optional[str] = None,
                 fortschritt=None) -> Dict[str, Any]:
    """Produces a complete consolidation plan. Read-only.

    The three models can be passed individually, so that the same body of
    memories can be planned with different models and the results compared
    afterwards (see vergleiche_plaene) - the same idea as in
    filter_compare.py, only for this task.
    """

    person = person or (USER_NAME if user_id == USER_ID else user_id.capitalize())
    group_model = group_model or SLEEP_GROUP_MODEL
    merge_model = merge_model or SLEEP_MERGE_MODEL
    age_model = age_model or SLEEP_AGE_MODEL
    usage_sammler: List = []

    if fortschritt:
        fortschritt("Loading memories")
    memories = load_memories(collection, user_id)

    aktionen: List[Dict[str, Any]] = []
    verbraucht: set = set()

    if 1 in stufen:
        if fortschritt:
            fortschritt("Stage 1: mechanical rules")
        neue = plan_mechanisch(memories, person)
        aktionen += neue
        for a in neue:
            verbraucht.update(q["id"] for q in a["quellen"])

    if 2 in stufen:
        neue = plan_verschmelzen(memories, verbraucht,
                                 group_model=group_model, merge_model=merge_model,
                                 usage_sammler=usage_sammler, fortschritt=fortschritt)
        aktionen += neue
        for a in neue:
            verbraucht.update(q["id"] for q in a["quellen"])

    if 3 in stufen:
        # Stage 3 is the one stage that depends entirely on a language
        # pack: without month and weekday names there is nothing to
        # pre-filter on. Say so rather than returning an empty stage that
        # looks like "nothing to consolidate".
        if not HEURISTIKEN_VERFUEGBAR and fortschritt:
            fortschritt(
                f"Stage 3 skipped: no language pack for "
                f"'{ASSISTANT_LANGUAGE}' (see language_packs.py)"
            )
        neue = plan_altern(memories, verbraucht, model=age_model,
                           usage_sammler=usage_sammler, fortschritt=fortschritt)
        aktionen += neue
        for a in neue:
            verbraucht.update(q["id"] for q in a["quellen"])

    # Number them through and set the pre-selection.
    for i, a in enumerate(aktionen, 1):
        a["nr"] = i
        a["angenommen"] = not a["pruefen"]

    kosten = sum(float(u.get("cost") or 0) for u in usage_sammler)
    token = sum(int(u.get("total_tokens") or 0) for u in usage_sammler)

    return {
        "erzeugt_am": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "person": person,
        "stufen": list(stufen),
        # Which language pack produced this plan. A plan made with the
        # German heuristics is not comparable to one made with the
        # English ones, so it belongs in the record.
        "sprache": SPRACHPAKET["code"] or ASSISTANT_LANGUAGE,
        # The models actually used, not the defaults - otherwise a later
        # comparison of two plans would be worthless.
        "modelle": {
            "grouping": group_model,
            "merging": merge_model,
            "ageing": age_model,
        },
        "anzahl_vorher": len(memories),
        "kosten_planung": kosten,
        "token_planung": token,
        "aktionen": aktionen,
    }


def speichere_plan(plan: Dict[str, Any]) -> str:
    os.makedirs(PLAN_DIR, exist_ok=True)
    ziel = os.path.join(
        PLAN_DIR, f"sleep_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(ziel, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    return ziel


def lade_plan(pfad: str) -> Dict[str, Any]:
    with open(pfad, encoding="utf-8") as f:
        return json.load(f)


def plan_kennzahlen(plan: Dict[str, Any]) -> Dict[str, int]:
    aktionen = plan.get("aktionen", [])
    angenommen = [a for a in aktionen if a.get("angenommen")]

    entfallend = sum(len(a["quellen"]) for a in angenommen)
    entstehend = sum(1 for a in angenommen if a.get("neu"))

    # In addition, the picture for the case that ALL actions are accepted.
    # Without that number one only reads the result of the pre-selection
    # off the screen and mistakes it for the effect of the run.
    alle_entfallend = sum(len(a["quellen"]) for a in aktionen)
    alle_entstehend = sum(1 for a in aktionen if a.get("neu"))
    vorher = plan.get("anzahl_vorher", 0)

    return {
        "aktionen_gesamt": len(aktionen),
        "aktionen_angenommen": len(angenommen),
        "zu_pruefen": sum(1 for a in aktionen if a.get("pruefen")),
        "erinnerungen_vorher": vorher,
        "quellen_entfallen": entfallend,
        "neue_eintraege": entstehend,
        "erinnerungen_nachher": vorher - entfallend + entstehend,
        "erinnerungen_nachher_alle": vorher - alle_entfallend + alle_entstehend,
    }


def vergleiche_plaene(plan_a: Dict[str, Any], plan_b: Dict[str, Any]) -> Dict[str, List]:
    """Contrasts two plans for the same body of memories.

    Intended for the question "is the more expensive model worth it?" - the
    same idea as filter_compare.py, only for consolidation. Two actions
    count as the same decision if they concern exactly the same source
    memories; what is then compared is whether the result matches.

    A model that consistently proposes fewer actions is not being careful -
    it finds less.
    """
    def schluessel(a):
        return frozenset(q["id"] for q in a["quellen"])

    karte_a = {schluessel(a): a for a in plan_a.get("aktionen", [])}
    karte_b = {schluessel(b): b for b in plan_b.get("aktionen", [])}

    nur_a = [karte_a[k] for k in karte_a.keys() - karte_b.keys()]
    nur_b = [karte_b[k] for k in karte_b.keys() - karte_a.keys()]

    gleich, verschieden = [], []
    for k in karte_a.keys() & karte_b.keys():
        a, b = karte_a[k], karte_b[k]
        text_a = (a.get("neu") or {}).get("text")
        text_b = (b.get("neu") or {}).get("text")
        (gleich if text_a == text_b else verschieden).append((a, b))

    return {
        "nur_a": sorted(nur_a, key=lambda x: x["nr"]),
        "nur_b": sorted(nur_b, key=lambda x: x["nr"]),
        "gleich": gleich,
        "verschieden": verschieden,
    }


# ==========================================================
# Applying
# ==========================================================

def sichere_datenbank() -> str:
    """Copies the complete chroma_db folder away.

    This is the real safety net - a JSON export would NOT be one, because
    memory_import.py loses the created_at values when playing the data
    back in (see the module docstring).
    """
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ziel = os.path.join(
        BACKUP_DIR, f"chroma_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copytree(CHROMA_DB_DIR, ziel)
    return ziel


def apply_plan(collection, plan: Dict[str, Any], mit_backup: bool = True,
               fortschritt=None, bloeckgroesse: int = None) -> Dict[str, Any]:
    """Applies the accepted actions of a plan.

    BATCHED INSTEAD OF ONE BY ONE
    -----------------------------
    Previously one collection.add() and one collection.delete() ran per
    action. With close to 200 actions that is around 400 individual Chroma
    transactions, each with its own SQLite commit and index touch.
    Measured against a body of 800 memories: 6.75 s one by one against
    0.38 s batched - a factor of 18, and the gap grows with the database.

    Batching happens in blocks, not in one single go. There are two
    reasons for that: progress stays visible, and an error costs at most
    one block instead of the whole run.

    ORDER
    -----
    Within each block, ALL new entries are created FIRST and the sources
    are deleted AFTERWARDS. If something aborts in between, both states
    exist - annoying, but repairable. The other way round, the facts would
    be irretrievably gone.

    PROGRESS
    --------
    fortschritt(text, anteil) is called once per block; anteil is a value
    between 0 and 1, or None. The user interface turns that into a progress
    bar, the command line ignores the second value.
    """
    user_id = plan["user_id"]
    angenommen = [a for a in plan.get("aktionen", []) if a.get("angenommen")]
    bloeckgroesse = bloeckgroesse or SLEEP_APPLY_BLOCK

    ergebnis = {
        "backup": None,
        "angelegt": 0,
        "geloescht": 0,
        "fehler": [],
        "dauer": 0.0,
    }

    if not angenommen:
        return ergebnis

    def melde(text, anteil=None):
        if fortschritt:
            try:
                fortschritt(text, anteil)
            except TypeError:
                # Older callbacks only know about the text.
                fortschritt(text)

    start = time.perf_counter()

    if mit_backup:
        melde("Backing up the database …", 0.0)
        try:
            ergebnis["backup"] = sichere_datenbank()
        except Exception as e:
            # WITHOUT a backup the run does not continue - the whole point
            # of the net is that it is stretched out before the jump.
            ergebnis["fehler"].append(f"Backup copy failed, aborting: {e}")
            return ergebnis

    dim = get_embedding_dimension(collection)
    jetzt = datetime.now(timezone.utc).isoformat()

    bloecke = [angenommen[i:i + bloeckgroesse]
               for i in range(0, len(angenommen), bloeckgroesse)]

    erledigt = 0

    for block_nr, block in enumerate(bloecke, 1):
        melde(
            f"Block {block_nr} of {len(bloecke)} – "
            f"action {erledigt + 1} to {erledigt + len(block)} of {len(angenommen)}",
            erledigt / len(angenommen),
        )

        neu_ids, neu_texte, neu_vektoren, neu_meta = [], [], [], []
        weg_ids: List[str] = []

        for a in block:
            quell_ids = [q["id"] for q in a["quellen"]]
            quell_daten = [q.get("created_at") for q in a["quellen"] if q.get("created_at")]

            if a.get("neu"):
                metadaten = {
                    "user_id": user_id,
                    "type": a["neu"]["type"],
                    "data": a["neu"]["text"],
                    "created_at": a.get("created_at") or jetzt,
                    "consolidated_at": jetzt,
                    "consolidated_stufe": a.get("stufe", ""),
                    # Chroma metadata only allows scalars - lists have to
                    # go in as a string.
                    "consolidated_from": ",".join(quell_ids),
                    "consolidated_count": len(quell_ids),
                }
                if quell_daten:
                    metadaten["original_created_at_min"] = min(quell_daten)

                neu_ids.append(str(uuid.uuid4()))
                neu_texte.append(a["neu"]["text"])
                neu_vektoren.append([0.0] * dim)
                neu_meta.append(metadaten)

            weg_ids.extend(quell_ids)

        nummern = ", ".join(str(a.get("nr", "?")) for a in block)

        try:
            if neu_ids:
                collection.add(ids=neu_ids, documents=neu_texte,
                               embeddings=neu_vektoren, metadatas=neu_meta)
                ergebnis["angelegt"] += len(neu_ids)
        except Exception as e:
            # Creating failed -> under no circumstances delete the sources.
            ergebnis["fehler"].append(
                f"Block {block_nr} (actions {nummern}): creating failed, "
                f"the sources are left untouched. {type(e).__name__}: {e}"
            )
            erledigt += len(block)
            continue

        try:
            if weg_ids:
                collection.delete(ids=weg_ids)
                ergebnis["geloescht"] += len(weg_ids)
        except Exception as e:
            ergebnis["fehler"].append(
                f"Block {block_nr} (actions {nummern}): deleting failed. "
                f"The new entries have already been created, so there are now "
                f"duplicates. {type(e).__name__}: {e}"
            )

        erledigt += len(block)

    ergebnis["dauer"] = time.perf_counter() - start
    melde(f"Finished in {ergebnis['dauer']:.1f}s", 1.0)

    # Write a log, so that what happened stays traceable even after a
    # window reload.
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        with open(os.path.join(BACKUP_DIR, "apply.log"), "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now().isoformat()}\t{len(angenommen)} actions\t"
                f"+{ergebnis['angelegt']} -{ergebnis['geloescht']}\t"
                f"{len(ergebnis['fehler'])} errors\t{ergebnis['dauer']:.1f}s\t"
                f"Backup: {ergebnis['backup']}\n"
            )
            for fehler in ergebnis["fehler"]:
                f.write(f"\tERROR: {fehler}\n")
    except Exception:
        pass

    return ergebnis


# ==========================================================
# Plain-text report
# ==========================================================

def report(plan: Dict[str, Any]) -> str:
    k = plan_kennzahlen(plan)
    m = plan.get("modelle", {})
    zeilen = [
        "=" * 78,
        "ALFRED SLEEP - CONSOLIDATION PLAN",
        "=" * 78,
        f"Created   : {plan.get('erzeugt_am', '?')}",
        f"User      : {plan.get('user_id')} (name used in texts: {plan.get('person')})",
        f"Stages    : {plan.get('stufen')}",
        f"Models    : grouping {m.get('grouping')} | "
        f"merging {m.get('merging')} | ageing {m.get('ageing')}",
        f"Cost      : ${plan.get('kosten_planung', 0):.5f} for {plan.get('token_planung', 0)} tokens",
        "",
        f"Memories before : {k['erinnerungen_vorher']}",
        f"Actions         : {k['aktionen_gesamt']} "
        f"({k['aktionen_angenommen']} accepted, {k['zu_pruefen']} to review)",
        f"Memories after  : {k['erinnerungen_nachher']} "
        f"(-{k['quellen_entfallen']} / +{k['neue_eintraege']})",
        "",
    ]

    for a in plan.get("aktionen", []):
        marke = "[x]" if a.get("angenommen") else "[ ]"
        pruef = "  <-- REVIEW" if a.get("pruefen") else ""
        zeilen.append("-" * 78)
        zeilen.append(
            f"{marke} #{a['nr']}  {a['aktion'].upper()}  "
            f"(stage: {a['stufe']}, confidence: {a.get('confidence', 1.0)}){pruef}"
        )
        for q in a["quellen"]:
            zeilen.append(f"      - [{q['type']}] {format_datum(q['created_at'])}: {q['text']}")
        if a.get("neu"):
            zeilen.append(f"    = [{a['neu']['type']}] {format_datum(a.get('created_at'))}: {a['neu']['text']}")
        else:
            zeilen.append("    = (will be deleted without replacement)")
        if a.get("begruendung"):
            zeilen.append(f"      Reason: {a['begruendung']}")

    zeilen.append("-" * 78)
    return "\n".join(zeilen)


# ==========================================================
# CLI
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Alfred Sleep - memory consolidation",
    )
    parser.add_argument("--user", default=USER_ID, help="user_id (default from config.py)")
    parser.add_argument("--stages", default="1,2,3",
                        help="Which stages should run, e.g. '1' or '1,2'")
    parser.add_argument("--apply", metavar="PLAN.JSON",
                        help="Apply an existing plan")
    parser.add_argument("--report", metavar="PLAN.JSON",
                        help="Print an existing plan as plain text")
    parser.add_argument("--no-backup", action="store_true",
                        help="Do NOT create a backup copy when applying (not recommended)")
    parser.add_argument("--group-model", default=None,
                        help=f"Model for the grouping (default: {SLEEP_GROUP_MODEL})")
    parser.add_argument("--merge-model", default=None,
                        help=f"Model for the merging (default: {SLEEP_MERGE_MODEL})")
    parser.add_argument("--age-model", default=None,
                        help=f"Model for the ageing (default: {SLEEP_AGE_MODEL})")
    parser.add_argument("--compare", nargs=2, metavar=("PLAN_A.JSON", "PLAN_B.JSON"),
                        help="Contrast two plans for the same body of memories")
    args = parser.parse_args()

    if args.compare:
        a, b = (lade_plan(p) for p in args.compare)
        v = vergleiche_plaene(a, b)
        ma = a.get("modelle", {}).get("merging", "A")
        mb = b.get("modelle", {}).get("merging", "B")

        print(f"A = {ma}  ({len(a.get('aktionen', []))} actions)")
        print(f"B = {mb}  ({len(b.get('aktionen', []))} actions)")
        print(f"\nSame decision, same result    : {len(v['gleich'])}")
        print(f"Same decision, other result   : {len(v['verschieden'])}")
        print(f"Only A found                  : {len(v['nur_a'])}")
        print(f"Only B found                  : {len(v['nur_b'])}")

        for titel, eintraege in (("ONLY A", v["nur_a"]), ("ONLY B", v["nur_b"])):
            if not eintraege:
                continue
            print(f"\n--- {titel} ---")
            for x in eintraege:
                ziel = (x.get("neu") or {}).get("text", "(delete)")
                print(f"  #{x['nr']} {x['aktion']}: {ziel[:70]}")

        if v["verschieden"]:
            print("\n--- DIFFERING RESULTS ---")
            for x, y in v["verschieden"]:
                print(f"  Sources: {len(x['quellen'])} memory/memories")
                print(f"    A: {(x.get('neu') or {}).get('text', '(delete)')[:70]}")
                print(f"    B: {(y.get('neu') or {}).get('text', '(delete)')[:70]}")
        return

    if args.report:
        print(report(lade_plan(args.report)))
        return

    collection = get_collection()

    if args.apply:
        plan = lade_plan(args.apply)
        k = plan_kennzahlen(plan)

        print(report(plan))
        print()
        print(f"{k['aktionen_angenommen']} action(s) will be applied.")
        print(f"{k['erinnerungen_vorher']} memories will become {k['erinnerungen_nachher']}.")

        if input("\nApply now? (yes/no): ").strip().lower() not in ("j", "ja", "y", "yes"):
            print("Aborted.")
            return

        ergebnis = apply_plan(
            collection, plan,
            mit_backup=not args.no_backup,
            fortschritt=lambda t, anteil=None: print(f"  {t}"),
        )

        print(f"\n{ergebnis['angelegt']} entry/entries created, "
              f"{ergebnis['geloescht']} deleted.")
        if ergebnis["backup"]:
            print(f"Backup: {ergebnis['backup']}")
        for f in ergebnis["fehler"]:
            print(f"  ERROR: {f}")
        return

    stufen = tuple(int(s) for s in args.stages.split(",") if s.strip().isdigit())

    plan = erzeuge_plan(collection, args.user, stufen=stufen,
                        group_model=args.group_model,
                        merge_model=args.merge_model,
                        age_model=args.age_model,
                        fortschritt=lambda t, anteil=None: print(f"  {t}"))
    pfad = speichere_plan(plan)

    print()
    print(report(plan))
    print()
    print(f"Plan saved      : {pfad}")
    print(f"Apply it with   : python alfred_sleep.py --apply \"{pfad}\"")
    print("Or more easily  : streamlit run alfred_sleep_ui.py")


if __name__ == "__main__":
    main()
