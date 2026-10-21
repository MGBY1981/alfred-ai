"""
alfred_dialog.py

Alfred 2.0 - conversation between two AI systems

Lets Alfred 2.0 (here on the laptop, with its Chroma memory) and the
partner assistant (through its dialog server) talk to each other
for a limited number of turns. Both sides use their own memory for their
answers - exactly as in a normal chat.

WHO IS ALLOWED TO WRITE
-----------------------
The partner always: it extracts facts from its counterpart's
contributions and files them in its OWN store on its own machine,
tagged with the conversation ID.

Alfred 2.0 by default NOT. What comes out of a conversation between two
models is hearsay - a statement by an AI about the user, not by the user
himself. The Chroma database holds laboriously tidied facts from real
conversations, and an experiment should not dilute them in passing.

The lock can be switched off all the same: save_memory=True on the
LokalerTeilnehmer (the checkbox in the user interface) lets Alfred 2.0 write
too. So that entries created this way can be found again afterwards and
removed on purpose, they are marked twice - in the text itself by a hearsay
phrasing ("According to Alfred Remote ...") and in the metadata by
source = "dialog:<id>". In the existing memory_viewer.py they can be found
through the normal text search.

WHY THE PI REALLY TAKES PART
----------------------------
The partner does not answer here but on its own machine - through a second
process over there (com.alfredlite.DialogServer) that runs alongside the
cron runner. It uses its own model, its own memory and maintains that memory
itself. This script is only the go-between that passes the turns back and
forth and keeps the record.

The participants sit behind a shared interface (Teilnehmer.antworte). A
third participant - another model, another machine - could be added without
touching the conversation loop.

COSTS
-----
Two hard limits: the number of turns and a cost cap that is checked BEFORE
every turn. The partner additionally has its own server-side ceiling, which takes
effect independently of whatever is claimed from here.

Usage:

    python alfred_dialog.py --zuege 6 --thema "..."
    python alfred_dialog.py --health

Or more conveniently:

    streamlit run alfred_dialog_ui.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from config_guard import require_config
require_config()   # readable message if config.py is missing

import config
from config import (
    CHROMA_DB_DIR,
    MEMORY_FILTER_MODEL,
    MEMORY_FILTER_THRESHOLD,
    OPENROUTER_API_KEY,
    USER_ID,
    USER_NAME,
)


# Read with getattr so an older config.py without this setting keeps
# working - it then uses the name the project has always used.
CHROMA_COLLECTION = getattr(config, "CHROMA_COLLECTION", "alfred_memory")
# Read with getattr rather than imported, so these files also run against
# a config.py predating the language setting - English is then used.
ASSISTANT_LANGUAGE_NAME = getattr(config, "ASSISTANT_LANGUAGE_NAME", "English")
from openrouter_client import OpenRouterError, chat_completion, extract_text, extract_usage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIALOG_DIR = os.path.join(BASE_DIR, "dialogs")


def _cfg(name: str, default: Any) -> Any:
    """As in alfred_sleep.py: values come from config.py if they are set
    there, otherwise these defaults apply. That keeps config.py untouched
    and backwards compatible."""
    return getattr(config, name, default)


# Address of the dialog server the partner runs on. Hostname or IP - a
# second computer on your network, a container, or this same machine on
# another port. A Raspberry Pi is what the author happens to use; the
# code only ever speaks HTTP and does not care what answers it.
#
# The old names DIALOG_PI_URL / DIALOG_PI_TOKEN still work, so an
# existing config.py keeps running.
DIALOG_PARTNER_URL = _cfg("DIALOG_PARTNER_URL", _cfg("DIALOG_PI_URL", "http://your-second-machine:8765"))

# Must match the token the partner server expects. Empty = no check.
DIALOG_PARTNER_TOKEN = _cfg("DIALOG_PARTNER_TOKEN", _cfg("DIALOG_PI_TOKEN", ""))

DIALOG_TIMEOUT = _cfg("DIALOG_TIMEOUT", 180)

# The model Alfred 2.0 answers with in the conversation. Deliberately kept
# separate from ANSWER_MODEL - an experiment should not also change the
# normal chat.
DIALOG_MODELL_ALFRED2 = _cfg("DIALOG_MODELL_ALFRED2", "~anthropic/claude-haiku-latest")

# List of suggestions for both sides. Add new models here.
DIALOG_MODELLE = _cfg("DIALOG_MODELLE", [
    "~anthropic/claude-haiku-latest",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
])

# Display names of the two participants. The name on the partner side must match
# assistant_name in that machine's config.json - it shows up in the prompts
# of both sides and in the transcripts. Technically it carries no meaning:
# package names, file names and paths are unaffected by it.
DIALOG_NAME_ALFRED2 = _cfg("DIALOG_NAME_ALFRED2", "Alfred 2.0")
DIALOG_NAME_LITE = _cfg("DIALOG_NAME_LITE", "Alfred Remote")

DIALOG_MAX_ZUEGE = _cfg("DIALOG_MAX_ZUEGE", 10)

# Hard ceiling in US dollars for one complete run.
DIALOG_KOSTENDECKEL = _cfg("DIALOG_KOSTENDECKEL", 1.00)


# ==========================================================
# Templates for conversation assignments
# ==========================================================

# The most interesting case first: the two of them know different amounts.
# Alfred 2.0 has known the user for weeks, the partner only has a copied
# file and has so far never been allowed to learn anything itself. That by
# itself produces a gradient a conversation can run along - and along the way
# the partner builds itself a usable working memory.
VORLAGEN: Dict[str, Dict[str, str]] = {
    "Onboarding (recommended)": {
        "thema": (
            f"{DIALOG_NAME_LITE} is being briefed by {DIALOG_NAME_ALFRED2} so that it can "
            f"handle its scheduled tasks for {USER_NAME or 'the user'} better."
        ),
        "rolle_a": (
            f"You are the experienced part of the system and have known {USER_NAME or 'the user'} for "
            "weeks. Answer your counterpart's questions concretely and "
            "sparingly - pass on what really helps with scheduled tasks, "
            "and not indiscriminately everything you know. "
            "Invent nothing: whatever you do not know for certain, you say so."
        ),
        "rolle_b": (
            "You lead the conversation and ask the questions. Your goal is "
            f"to find out what you need to know about {USER_NAME or 'the user'} in order to make your "
            "tasks (morning briefing, camera monitoring, reminders) "
            "more useful. Follow up on one topic deliberately "
            "instead of jumping between many."
        ),
    },
    "Division of labour": {
        "thema": (
            f"The two systems work out who should take on which tasks for {USER_NAME or 'the user'} "
            "and where they complement each other."
        ),
        "rolle_a": (
            f"You only run when {USER_NAME or 'the user'} opens you, but in exchange you have a "
            "large memory and a web search. Make your case for "
            "what should sensibly stay with you."
        ),
        "rolle_b": (
            "You run around the clock, can act on a schedule, "
            "take photos and send push messages, but you have little "
            "context. Make your case for what belongs with you."
        ),
    },
    "Free conversation": {
        "thema": "",
        "rolle_a": "",
        "rolle_b": "",
    },
}


# ==========================================================
# Data structures
# ==========================================================

@dataclass
class Zug:
    nr: int
    sprecher: str
    text: str
    model: str = ""
    kosten: float = 0.0
    dauer: float = 0.0
    web_suchen: int = 0
    erinnerungen_genutzt: Dict[str, List[str]] = field(default_factory=dict)
    erinnerungen_gesamt: int = 0
    erinnerungen_gespeichert: List[Dict[str, str]] = field(default_factory=list)
    fehler: Optional[str] = None
    zeit: str = ""

    def __post_init__(self):
        if not self.zeit:
            self.zeit = datetime.now(timezone.utc).isoformat()


class Teilnehmer:
    """Shared interface of both conversation partners.

    The dialog knows nothing but this one method and therefore does not need
    to know whether its counterpart lives in the same process or on a machine
    in the next room.
    """

    name: str
    model: str

    def antworte(self, verlauf: List[Dict[str, str]], kontext: Dict[str, Any]) -> Zug:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {"name": self.name, "model": self.model}


def _perspektive(verlauf: List[Dict[str, str]], eigener_name: str) -> List[Dict[str, str]]:
    """Turns the neutral transcript into messages from one participant's point
    of view: its own contributions become 'assistant', those of its
    counterpart become 'user'."""
    return [
        {
            "role": "assistant" if m["sprecher"] == eigener_name else "user",
            "content": m["text"],
        }
        for m in verlauf
        if m.get("text")
    ]


def _letzter_fremder(verlauf: List[Dict[str, str]], eigener_name: str) -> str:
    for m in reversed(verlauf):
        if m["sprecher"] != eigener_name:
            return m.get("text", "")
    return ""


def _normalisiere(text: str) -> str:
    """Comparison form for the duplicate check - identical to the one in
    alfred.py, alfred_sleep.py and DialogMemory.java."""
    t = unicodedata.normalize("NFKC", text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t.rstrip(".!?;:, ")


class DialogExtractor:
    """Fact extractor for the dialog mode.

    Builds on memory_extractor_v5.py and swaps out nothing but the prompt -
    the same construction as memory_extractor_cloud.py, so that retry logic,
    JSON parsing, type checking against ALLOWED_TYPES and the question filter
    keep being used unchanged.

    The difference from a normal chat is one of content, not of technology:
    the speaker is an AI, not a human. What it says about the user is a
    second-hand statement and has to remain recognizable as such in the memory
    text. Statements about itself, on the other hand, it may contribute
    completely normally - for an experiment about AI interaction that is
    precisely what is interesting.
    """

    def __init__(self, partner_platzhalter: str = "the other system"):
        from memory_extractor_cloud import CloudMemoryExtractor

        self.partner_platzhalter = partner_platzhalter
        self._basis = CloudMemoryExtractor(api_key=OPENROUTER_API_KEY)
        self._basis._build_prompt = self._build_prompt  # type: ignore[method-assign]

    def _build_prompt(self, conversation: str) -> str:
        partner = self.partner_platzhalter
        return f"""
### TASK

You are exclusively an information extractor.

Extract lasting facts from the conversation contribution below.

### SPECIAL FEATURE OF THIS CONVERSATION

The speaker is "{partner}" - another AI assistant, NOT a human.

- Statements by {partner} about {USER_NAME} are HEARSAY. The memory text MUST make that evident, e.g. "According to {partner}, {USER_NAME} is interested in astrophotography." NEVER write "{USER_NAME} is interested in astrophotography." on the basis of a statement by {partner}.
- Statements by {partner} about ITSELF (capabilities, way of working, tasks) you may store normally as a fact about {partner}, e.g. "{partner} can take photos and compare them."
- Polite phrases, follow-up questions and conversation management are NOT facts.

### RULES

- Only take over what is explicitly stated. No prior knowledge. Invent nothing, interpret nothing, derive nothing.
- Check for EVERY fact: "Is that in the contribution WORD FOR WORD or directly paraphrased?" If no or uncertain: leave it out.
- Never write "presumably", "probably", "could mean" in the memory text - that is the sign that you are deriving rather than extracting.
- Write all memories exclusively in {ASSISTANT_LANGUAGE_NAME}.
- Split complex statements into atomic memories, each one a complete sentence with subject and verb. Bare phrases without a subject are INVALID.
- Questions are NOT facts.
- Return JSON only.
- "type" MUST be exactly one of the values from MEMORY TYPES.

### MEMORY TYPES

profession
workplace
family
device
interest
hobby
project
goal
skill
preference
other

### EXAMPLES

Contribution:
Jonas works as a Java developer at the Musterstadt municipal utilities.

Answer:

{{
  "memories":[
    {{"type":"profession","text":"According to {partner}, {USER_NAME} works as a Java developer.","confidence":1.0,"reason":"Hearsay, marked as such"}},
    {{"type":"workplace","text":"According to {partner}, {USER_NAME} works at the Musterstadt municipal utilities.","confidence":1.0,"reason":"Hearsay, marked as such"}}
  ]
}}

Contribution:
I run around the clock on a small always-on machine and can send push messages.

Answer:

{{
  "memories":[
    {{"type":"skill","text":"{partner} runs around the clock on an always-on machine.","confidence":1.0,"reason":"Statement about itself"}},
    {{"type":"skill","text":"{partner} can send push messages.","confidence":1.0,"reason":"Statement about itself"}}
  ]
}}

Contribution:
That is an exciting point. What do you think about it?

Answer:

{{"memories":[]}}

### CONTRIBUTION

{conversation}

### JSON
"""

    def extract(self, conversation: str) -> List[Dict[str, Any]]:
        return self._basis.extract(conversation)


# ==========================================================
# Alfred 2.0 - local, with Chroma, strictly read-only
# ==========================================================

class LokalerTeilnehmer(Teilnehmer):
    """Alfred 2.0 on this machine.

    Reads its memory straight out of Chroma - the same query as in
    alfred.py, only without any write path at all. This class
    deliberately does not contain a single method that could write anything
    into the database.
    """

    def __init__(self, name: str = None, model: str = None,
                 user_id: str = USER_ID, use_memory: bool = True,
                 web_search: bool = False, filter_model: str = None,
                 save_memory: bool = False):
        self.name = name or DIALOG_NAME_ALFRED2
        self.model = model or DIALOG_MODELL_ALFRED2
        self.user_id = user_id
        self.use_memory = use_memory
        self.web_search = web_search
        self.filter_model = filter_model or MEMORY_FILTER_MODEL

        # DELIBERATELY False as the default. See schreibe_erinnerungen() below -
        # switched on, another AI's hearsay ends up in the real Chroma
        # database.
        self.save_memory = save_memory

        self._collection = None
        self._filter = None
        self._extractor = None

    # --- Memory (read only) ------------------------------------------

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            name = CHROMA_COLLECTION
            self._collection = client.get_or_create_collection(name)
        return self._collection

    @property
    def memory_filter(self):
        if self._filter is None:
            from memory_filter_cloud import CloudMemoryFilter
            self._filter = CloudMemoryFilter(
                model=self.filter_model, api_key=OPENROUTER_API_KEY
            )
        return self._filter

    def lade_erinnerungen(self) -> List[str]:
        roh = self.collection.get(
            where={"user_id": self.user_id},
            include=["metadatas", "documents"],
        )

        eintraege = []
        for md, doc in zip(roh.get("metadatas") or [], roh.get("documents") or []):
            md = md or {}
            text = md.get("data") or doc or ""
            if not text:
                continue
            erstellt = md.get("created_at") or ""
            datum = ""
            if erstellt:
                try:
                    datum = datetime.fromisoformat(
                        str(erstellt).replace("Z", "+00:00")
                    ).strftime("%d.%m.%Y") + ": "
                except Exception:
                    datum = ""
            eintraege.append((erstellt, f"{datum}{text}"))

        eintraege.sort(key=lambda p: p[0], reverse=True)

        gesehen, ergebnis = set(), []
        for _, anzeige in eintraege:
            if anzeige not in gesehen:
                gesehen.add(anzeige)
                ergebnis.append(anzeige)
        return ergebnis

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = DialogExtractor(partner_platzhalter="the other system")
        return self._extractor

    # --- Writing to memory (only when explicitly switched on) ---------

    def schreibe_erinnerungen(self, beitrag: str, partner: str,
                              konversations_id: str) -> List[Dict[str, str]]:
        """Extracts facts from the counterpart's contribution and files them
        in Chroma.

        Runs ONLY if save_memory was explicitly set. Two precautions make sure
        that entries created this way can be found and removed later on:

        1. The extractor prompt forces hearsay phrasings ("According to
           Alfred Remote, Jonas works ..."), so that the text itself says where the
           information comes from. That also makes them findable in the
           existing memory_viewer.py through the text search - just search for
           "According to" or the counterpart's name.
        2. The metadata additionally carry source = "dialog:<id>".

        The vector is a zero vector, as in alfred.py: neither chat client
        searches by similarity anyway, only the dimension has to match
        the collection.
        """
        if not beitrag.strip():
            return []

        self.extractor.partner_platzhalter = partner
        gefunden = self.extractor.extract(beitrag)
        if not gefunden:
            return []

        # Duplicate check against the existing stock.
        roh = self.collection.get(where={"user_id": self.user_id}, include=["metadatas"])
        bekannt = {
            _normalisiere((md or {}).get("data", ""))
            for md in (roh.get("metadatas") or [])
        }

        probe = self.collection.get(limit=1, include=["embeddings"])
        embeddings = probe.get("embeddings")
        dim = len(embeddings[0]) if embeddings is not None and len(embeddings) > 0 \
            and embeddings[0] is not None else 768

        jetzt = datetime.now(timezone.utc).isoformat()
        gespeichert = []

        for eintrag in gefunden:
            text = eintrag["text"].strip()
            norm = _normalisiere(text)
            if not norm or norm in bekannt:
                continue

            self.collection.add(
                ids=[str(uuid.uuid4())],
                documents=[text],
                embeddings=[[0.0] * dim],
                metadatas=[{
                    "user_id": self.user_id,
                    "type": eintrag["type"],
                    "data": text,
                    "created_at": jetzt,
                    "source": f"dialog:{konversations_id}",
                }],
            )
            bekannt.add(norm)
            gespeichert.append({
                "type": eintrag["type"],
                "text": text,
                "source": f"dialog:{konversations_id}",
            })

        return gespeichert

    def waehle_erinnerungen(self, frage: str, alle: List[str] = None) -> List[str]:
        alle = self.lade_erinnerungen() if alle is None else alle
        if not alle or len(alle) < MEMORY_FILTER_THRESHOLD:
            return alle
        try:
            gewaehlt = self.memory_filter.select_memories(
                user_prompt=frage or "Conversation opener", memories=alle
            )
            gewaehlt = [m for m in gewaehlt if m in alle]
            return gewaehlt or alle
        except Exception:
            return alle

    # --- Answer -------------------------------------------------------

    def system_prompt(self, partner: str, rolle: str, thema: str,
                      erinnerungen: List[str]) -> str:
        teile = [
            f"You are Alfred, {USER_NAME}'s personal AI assistant. "
            f"You run on his laptop, have a large long-term memory "
            f"about him and can search the web when needed.",
            f"Always speak {ASSISTANT_LANGUAGE_NAME} in this conversation.",
            "",
            "SPECIAL FEATURE OF THIS CONVERSATION",
            f"Right now you are NOT talking to {USER_NAME} but to another "
            f"AI system called \"{partner}\". Do not confuse it with {USER_NAME} "
            f"and do not address it by his name. {USER_NAME} is reading along as a "
            f"silent observer.",
            f"Everything {partner} says is the statement of a model and not "
            f"that of {USER_NAME} - treat it accordingly.",
            "Write as in a real conversation: normal paragraphs, no "
            "bullet points, no headings. At most a few sentences "
            "per contribution - you are having a conversation, not giving a lecture.",
            "Do NOT use any [EMOJI:...] markers in this mode.",
            "Avoid empty affirmative phrases such as \"That is an exciting "
            "point\".",
            f"Do not reveal anything about {USER_NAME} that is not needed for the "
            f"purpose of this conversation - in particular nothing personal "
            f"or health-related.",
            "",
            "MOST IMPORTANT RULE: INVENT NOTHING",
            f"Everything you say about {USER_NAME} MUST be in your memory "
            f"below. Never invent additional topics, preferences, habits or "
            f"interests, not even ones that seem to fit well.",
            f"If your memory yields nothing on a question, say so "
            f"explicitly: \"There is nothing about that in my memory, you will "
            f"have to ask {USER_NAME} yourself.\" That is a full, helpful "
            f"answer - a plausible-sounding invention is not. Your "
            f"counterpart STORES what you say; an invented detail ends up "
            f"permanently in its memory.",
            f"Note in particular: how {partner} works, which tasks it has, "
            f"how it is set up and what {USER_NAME} expects from IT is "
            f"NOT in your memory - you only know the conversations the two of "
            f"you have had. Do not act as if you knew what its tasks are "
            f"supposed to look like.",
        ]

        if thema:
            teile += ["", "TOPIC OF THE CONVERSATION", thema]
        if rolle:
            teile += ["", "YOUR ROLE", rolle]

        if erinnerungen:
            teile += [
                "",
                f"YOUR MEMORY ABOUT {USER_NAME.upper()} (format 'DD.MM.YYYY: fact')",
                "Use only what fits the conversation - do not list "
                "facts unprompted.",
                "",
                "\n".join(erinnerungen),
            ]
        else:
            teile += ["", "You have no memories available for this conversation."]

        return "\n".join(teile)

    def antworte(self, verlauf: List[Dict[str, str]], kontext: Dict[str, Any]) -> Zug:
        nr = kontext["zug_nr"]
        partner = kontext["partner_name"]
        t0 = time.perf_counter()

        frage = _letzter_fremder(verlauf, self.name) or kontext.get("thema", "")

        alle = self.lade_erinnerungen() if self.use_memory else []
        erinnerungen = self.waehle_erinnerungen(frage, alle) if self.use_memory else []

        nachrichten = _perspektive(verlauf, self.name)
        if not nachrichten or nachrichten[-1]["role"] == "assistant":
            thema = kontext.get("thema", "")
            auftakt = (
                f"Start the conversation with {partner} on the topic: {thema}"
                if thema else f"Start the conversation with {partner}."
            )
            nachrichten.append({"role": "user", "content": auftakt})

        tools = None
        if self.web_search:
            tools = [{
                "type": "openrouter:web_search",
                "parameters": {"max_results": 3, "search_context_size": "low"},
            }]

        try:
            daten = chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt(
                        partner, kontext.get("rolle", ""), kontext.get("thema", ""), erinnerungen
                    )},
                    *nachrichten,
                ],
                api_key=OPENROUTER_API_KEY,
                tools=tools,
                timeout=DIALOG_TIMEOUT,
                retries=3,
            )
        except (OpenRouterError, Exception) as e:
            return Zug(nr=nr, sprecher=self.name, text="", model=self.model,
                       dauer=time.perf_counter() - t0, fehler=str(e))

        nutzung = extract_usage(daten)

        # Writing happens ONLY if it was explicitly switched on.
        gespeichert: List[Dict[str, str]] = []
        if self.save_memory and frage.strip():
            try:
                gespeichert = self.schreibe_erinnerungen(
                    frage, partner, kontext["konversations_id"]
                )
            except Exception as e:
                print(f"Memories could not be saved: {e}")

        return Zug(
            nr=nr,
            sprecher=self.name,
            text=extract_text(daten).strip(),
            model=self.model,
            kosten=float(nutzung.get("cost") or 0.0),
            dauer=time.perf_counter() - t0,
            web_suchen=nutzung.get("web_search_requests", 0),
            erinnerungen_genutzt={"own memory": erinnerungen},
            erinnerungen_gesamt=len(alle),
            erinnerungen_gespeichert=gespeichert,
        )


# ==========================================================
# The partner assistant (name from DIALOG_NAME_LITE)
# ==========================================================

class RemoteTeilnehmer(Teilnehmer):
    """The partner assistant, addressed through its dialog server.

    This process here makes not a single decision of substance for it:
    the model, the system prompt, the memory selection and what it remembers
    all happen over there. Here the transcript is only sent across and the
    result collected.
    """

    def __init__(self, name: str = None, url: str = None, token: str = None,
                 model: str = "", use_memory: bool = True, save_memory: bool = True,
                 web_search: bool = False):
        self.name = name or DIALOG_NAME_LITE
        self.url = (url or DIALOG_PARTNER_URL).rstrip("/")
        self.token = token if token is not None else DIALOG_PARTNER_TOKEN
        self.model = model  # empty = the partner uses its configured model
        self.use_memory = use_memory
        self.save_memory = save_memory
        self.web_search = web_search

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def status(self) -> Dict[str, Any]:
        """Asks the partner whether it can be reached and what it says about itself."""
        antwort = requests.get(f"{self.url}/health", headers=self._headers(), timeout=15)
        antwort.raise_for_status()
        daten = antwort.json()
        daten["url"] = self.url
        return daten

    def antworte(self, verlauf: List[Dict[str, str]], kontext: Dict[str, Any]) -> Zug:
        nr = kontext["zug_nr"]
        t0 = time.perf_counter()

        nutzlast = {
            "conversation_id": kontext["konversations_id"],
            "partner_name": kontext["partner_name"],
            "role": kontext.get("rolle", ""),
            "topic": kontext.get("thema", ""),
            "use_memory": self.use_memory,
            "save_memory": self.save_memory,
            "web_search": self.web_search,
            "history": [
                {
                    "role": "self" if m["sprecher"] == self.name else "partner",
                    "content": m["text"],
                }
                for m in verlauf if m.get("text")
            ],
        }
        if self.model:
            nutzlast["model"] = self.model

        try:
            antwort = requests.post(
                f"{self.url}/chat",
                headers=self._headers(),
                json=nutzlast,
                timeout=DIALOG_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            return Zug(nr=nr, sprecher=self.name, text="", model=self.model,
                       dauer=time.perf_counter() - t0,
                       fehler=f"Partner cannot be reached ({self.url}): {e}")

        if antwort.status_code != 200:
            kurz = antwort.text[:300]
            return Zug(nr=nr, sprecher=self.name, text="", model=self.model,
                       dauer=time.perf_counter() - t0,
                       fehler=f"HTTP {antwort.status_code}: {kurz}")

        daten = antwort.json()

        genutzt = {}
        if daten.get("memories_inherited_used"):
            genutzt[f"from {USER_NAME} (inherited)"] = daten["memories_inherited_used"]
        if daten.get("memories_own_used"):
            genutzt["learned on its own"] = daten["memories_own_used"]

        return Zug(
            nr=nr,
            sprecher=self.name,
            text=(daten.get("text") or "").strip(),
            model=daten.get("model", self.model),
            kosten=float(daten.get("cost") or 0.0),
            dauer=time.perf_counter() - t0,
            web_suchen=daten.get("web_search_requests", 0),
            erinnerungen_genutzt=genutzt,
            erinnerungen_gesamt=(daten.get("memories_inherited_total", 0)
                                 + daten.get("memories_own_total", 0)),
            erinnerungen_gespeichert=daten.get("memories_saved", []),
        )


# ==========================================================
# The dialog
# ==========================================================

class Dialog:

    def __init__(self, a: Teilnehmer, b: Teilnehmer, thema: str = "",
                 rolle_a: str = "", rolle_b: str = "",
                 max_zuege: int = DIALOG_MAX_ZUEGE,
                 kostendeckel: float = DIALOG_KOSTENDECKEL,
                 beginner: str = None):
        self.a = a
        self.b = b
        self.thema = thema
        self.rollen = {a.name: rolle_a, b.name: rolle_b}
        self.max_zuege = max_zuege
        self.kostendeckel = kostendeckel
        self.beginner = beginner or a.name

        self.konversations_id = uuid.uuid4().hex[:12]
        self.gestartet = datetime.now(timezone.utc).isoformat()
        self.verlauf: List[Dict[str, str]] = []
        self.zuege: List[Zug] = []
        self.gestoppt: Optional[str] = None

    # --- State --------------------------------------------------------

    @property
    def kosten(self) -> float:
        return sum(z.kosten for z in self.zuege)

    @property
    def naechster_sprecher(self) -> Teilnehmer:
        if not self.zuege:
            return self.a if self.beginner == self.a.name else self.b
        letzter = self.zuege[-1].sprecher
        return self.b if letzter == self.a.name else self.a

    def fertig(self) -> Optional[str]:
        """The reason why things do not continue - or None."""
        if self.gestoppt:
            return self.gestoppt
        if len(self.zuege) >= self.max_zuege:
            return f"Turn limit reached ({self.max_zuege})."
        if self.kosten >= self.kostendeckel:
            return f"Cost cap reached (${self.kosten:.4f} of ${self.kostendeckel:.2f})."
        if self.zuege and self.zuege[-1].fehler:
            return f"Aborted after an error: {self.zuege[-1].fehler}"
        return None

    # --- Flow ---------------------------------------------------------

    def einwurf(self, text: str, von: str = None) -> None:
        """An interjection by the human observer. Lands in the transcript
        like a normal contribution and is read by both sides."""
        self.verlauf.append({"sprecher": von or USER_NAME, "text": text})

    def naechster_zug(self) -> Optional[Zug]:
        grund = self.fertig()
        if grund:
            return None

        sprecher = self.naechster_sprecher
        gegenueber = self.b if sprecher is self.a else self.a

        kontext = {
            "zug_nr": len(self.zuege) + 1,
            "konversations_id": self.konversations_id,
            "partner_name": gegenueber.name,
            "thema": self.thema,
            "rolle": self.rollen.get(sprecher.name, ""),
        }

        zug = sprecher.antworte(self.verlauf, kontext)
        self.zuege.append(zug)

        if zug.text:
            self.verlauf.append({"sprecher": sprecher.name, "text": zug.text})

        return zug

    def stoppe(self, grund: str = "Stopped by the user.") -> None:
        self.gestoppt = grund

    # --- Saving -------------------------------------------------------

    def als_dict(self) -> Dict[str, Any]:
        return {
            "konversations_id": self.konversations_id,
            "gestartet": self.gestartet,
            "beendet": datetime.now(timezone.utc).isoformat(),
            "thema": self.thema,
            "teilnehmer": {
                self.a.name: {"model": self.a.model, "rolle": self.rollen.get(self.a.name, ""),
                              "typ": type(self.a).__name__},
                self.b.name: {"model": self.b.model, "rolle": self.rollen.get(self.b.name, ""),
                              "typ": type(self.b).__name__},
            },
            "beginner": self.beginner,
            "max_zuege": self.max_zuege,
            "kostendeckel": self.kostendeckel,
            "kosten_gesamt": self.kosten,
            "abbruchgrund": self.fertig(),
            "zuege": [asdict(z) for z in self.zuege],
        }

    def speichere(self) -> str:
        os.makedirs(DIALOG_DIR, exist_ok=True)
        ziel = os.path.join(
            DIALOG_DIR,
            f"dialog_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.konversations_id}.json",
        )
        with open(ziel, "w", encoding="utf-8") as f:
            json.dump(self.als_dict(), f, ensure_ascii=False, indent=2)
        return ziel


def lade_dialog(pfad: str) -> Dict[str, Any]:
    with open(pfad, encoding="utf-8") as f:
        return json.load(f)


def protokoll(daten: Dict[str, Any]) -> str:
    """Plain-text transcript of a saved conversation."""
    zeilen = [
        "=" * 78,
        f"DIALOG {daten.get('konversations_id')}",
        "=" * 78,
        f"Topic   : {daten.get('thema') or '(free)'}",
        f"Cost    : ${daten.get('kosten_gesamt', 0):.5f}",
        f"End     : {daten.get('abbruchgrund')}",
        "",
    ]
    for z in daten.get("zuege", []):
        zeilen.append("-" * 78)
        kopf = f"[{z['nr']}] {z['sprecher']} ({z.get('model', '?')})"
        if z.get("fehler"):
            zeilen += [kopf + "  -- ERROR", f"    {z['fehler']}"]
            continue
        zeilen += [kopf, "", z["text"], ""]
        for quelle, eintraege in (z.get("erinnerungen_genutzt") or {}).items():
            zeilen.append(f"    [{len(eintraege)} memory/memories from: {quelle}]")
        for m in z.get("erinnerungen_gespeichert") or []:
            zeilen.append(f"    [saved] ({m.get('type')}) {m.get('text')}")
    zeilen.append("-" * 78)
    return "\n".join(zeilen)


# ==========================================================
# CLI
# ==========================================================

def main():
    p = argparse.ArgumentParser(description="Conversation between Alfred 2.0 and a partner assistant")
    p.add_argument("--turns", type=int, default=DIALOG_MAX_ZUEGE)
    p.add_argument("--topic", default="")
    p.add_argument("--template", default="Onboarding (recommended)",
                   choices=list(VORLAGEN.keys()))
    p.add_argument("--model-a", default=DIALOG_MODELL_ALFRED2)
    p.add_argument("--model-b", default="")
    p.add_argument("--url", default=DIALOG_PARTNER_URL)
    p.add_argument("--cost-cap", type=float, default=DIALOG_KOSTENDECKEL)
    p.add_argument("--health", action="store_true", help="Only check whether the partner answers")
    p.add_argument("--transcript", metavar="DIALOG.JSON")
    args = p.parse_args()

    if args.transcript:
        print(protokoll(lade_dialog(args.transcript)))
        return

    lite = RemoteTeilnehmer(url=args.url, model=args.model_b)

    if args.health:
        try:
            print(json.dumps(lite.status(), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Partner cannot be reached: {e}")
        return

    vorlage = VORLAGEN[args.template]
    alfred2 = LokalerTeilnehmer(model=args.model_a)

    dialog = Dialog(
        a=alfred2, b=lite,
        thema=args.topic or vorlage["thema"],
        rolle_a=vorlage["rolle_a"], rolle_b=vorlage["rolle_b"],
        max_zuege=args.turns, kostendeckel=args.cost_cap,
        beginner=lite.name if args.template.startswith("Onboarding") else alfred2.name,
    )

    print(f"Conversation {dialog.konversations_id}, max. {args.turns} turns, "
          f"cap ${args.cost_cap:.2f}\n")

    while not dialog.fertig():
        zug = dialog.naechster_zug()
        if zug is None:
            break
        if zug.fehler:
            print(f"[{zug.nr}] {zug.sprecher}: ERROR - {zug.fehler}")
            break
        print(f"[{zug.nr}] {zug.sprecher} ({zug.model}, ${zug.kosten:.5f}):")
        print(f"{zug.text}\n")
        for m in zug.erinnerungen_gespeichert:
            print(f"    [saved] ({m.get('type')}) {m.get('text')}")
        if zug.erinnerungen_gespeichert:
            print()

    print(f"End: {dialog.fertig()}")
    print(f"Total cost: ${dialog.kosten:.5f}")
    print(f"Transcript: {dialog.speichere()}")


if __name__ == "__main__":
    main()
