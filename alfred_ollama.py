"""
alfred_ollama.py

Alfred 2.0 - the chat client, variant with mem0 and local Ollama.

Functionally the same assistant as alfred.py: same memory, same
database, same config.py, same tools around it, 13 shared functions. The
difference is one single thing - the write path.

    alfred.py          writes to Chroma directly. Needs an OpenRouter
                       key and nothing else. The recommended default.

    alfred_ollama.py   writes through mem0, which produces a real
                       embedding vector for every memory and needs a
                       local Ollama for that (nomic-embed-text).

WHICH ONE DO YOU WANT?
----------------------
Take alfred.py unless you have a specific reason not to. It is simpler
to set up and the whole chat runs without a local AI installation.

Take this one if you want mem0 in the write path - for instance because
you want real embedding vectors in the database for something else, or
because you want to build on mem0's own features later.

REQUIREMENTS
------------
Ollama has to be installed AND running, with the embedding model
pulled:

    ollama pull nomic-embed-text

Without a reachable Ollama this client cannot start - mem0 raises on
initialisation. That is not a bug, it is the point of this variant;
alfred.py is the one that works without it.

NOTE ON THE DATABASE
--------------------
Both variants write into the same collection, which is deliberate - you
can switch between them at will. It does mean the vectors in the
database end up mixed: real ones from here, zero vectors from
alfred.py. Nothing searches by similarity, so this does no harm, but it
is the reason alfred_sleep.py groups memories by model judgement rather
than by vector distance.

Start:  streamlit run alfred_ollama.py   (or start_alfred_ollama.bat)
"""

import os
import re
import time
import unicodedata
from datetime import datetime

import streamlit as st

os.environ.setdefault("MEM0_TELEMETRY", "False")  # see memory_import.py

from mem0 import Memory

from config_guard import require_config
require_config()   # readable message if config.py is missing

import config
from config import (
    ANSWER_MODEL,
    ANSWER_TIMEOUT,
    ANSWER_WEB_SEARCH_CONTEXT_SIZE,
    ANSWER_WEB_SEARCH_MAX_RESULTS,
    EMOJI_DIR,
    MAX_CHAT_HISTORY_MESSAGES,
    MEM0_CONFIG,
    MEMORY_EXTRACTION_DEFERRED,
    MEMORY_EXTRACTOR_BACKEND,
    MEMORY_EXTRACTOR_CLOUD_MODEL,
    MEMORY_EXTRACTOR_MODEL,
    MEMORY_FILTER_MODE,
    MEMORY_FILTER_MODEL,
    MEMORY_FILTER_THRESHOLD,
    MEMORY_MANAGER_MODEL,
    OLLAMA_HOST,
    OLLAMA_NUM_THREAD,
    OPENROUTER_API_KEY,
    SHOW_TIMINGS,
    USER_LOCATION,
    USER_NAME,
)


# Read with getattr so an older config.py without this setting keeps
# working - it then uses the name the project has always used.
CHROMA_COLLECTION = getattr(config, "CHROMA_COLLECTION", "alfred_memory")
# Read with getattr rather than imported, so these files also run against
# a config.py predating the language setting - English is then used.
ASSISTANT_LANGUAGE_NAME = getattr(config, "ASSISTANT_LANGUAGE_NAME", "English")
from openrouter_client import (
    OpenRouterError,
    chat_completion,
    extract_text,
    extract_usage,
)

# set_page_config MUST be the first Streamlit command in the script.
st.set_page_config(page_title="Alfred 2.0", page_icon="🤖")


# ==========================================================
# Initialisation (each one exactly once per Streamlit process)
# ==========================================================

@st.cache_resource
def init_memory():
    try:
        return Memory.from_config(MEM0_CONFIG)
    except Exception as e:
        # This variant stands or falls with a running Ollama - see the
        # file header. A red traceback would not tell a newcomer that,
        # and alfred.py next door needs none of it.
        st.error(
            "**This variant cannot reach Ollama.**  \n\n"
            "`alfred_ollama.py` stores through mem0, which needs a "
            "running Ollama with `ollama pull nomic-embed-text`.  \n\n"
            "If you did not mean to take the local route, start "
            "`alfred.py` instead - same assistant, same database, and it "
            f"needs nothing but your OpenRouter key.  \n\n*{e}*",
            icon="🦙",
        )
        st.stop()


@st.cache_resource
def init_chroma_collection(_memory):
    """Accesses the Chroma collection that mem0 has already created
    INTERNALLY while building the Memory instance, instead of creating a
    new chromadb.PersistentClient of our own. A second, independent
    client for the same path would crash Chroma with
    'An instance of Chroma already exists ... with different settings',
    because Chroma allows only one client process per path.

    _memory with a leading underscore so that Streamlit does not try to
    hash the mem0 object as a cache key (it is not hashable).
    """
    vector_store = getattr(_memory, "vector_store", None)
    if vector_store is None:
        raise RuntimeError("Could not read 'vector_store' from the mem0 Memory instance.")

    collection = getattr(vector_store, "collection", None)
    if collection is not None:
        return collection

    client = getattr(vector_store, "client", None)
    if client is not None:
        collection_name = CHROMA_COLLECTION
        return client.get_collection(collection_name)

    raise RuntimeError(
        "Could not read the Chroma collection from mem0's internal vector_store "
        "(neither .collection nor .client found - possibly a different mem0 version)."
    )


@st.cache_resource
def init_memory_filter():
    """Returns a filter or None, depending on the configuration.

    Both variants have the same method select_memories(user_prompt,
    memories) -> List[str], so the caller does not have to tell them
    apart.
    """
    if MEMORY_FILTER_MODE == "openrouter":
        from memory_filter_cloud import CloudMemoryFilter
        return CloudMemoryFilter(model=MEMORY_FILTER_MODEL, api_key=OPENROUTER_API_KEY)

    if MEMORY_FILTER_MODE == "ollama":
        from memory_manager_v2_1 import MemoryManager
        return MemoryManager(
            model=MEMORY_MANAGER_MODEL,
            host=OLLAMA_HOST,
            num_thread=OLLAMA_NUM_THREAD,
        )

    return None


@st.cache_resource
def init_memory_extractor():
    """Returns an extractor or None, depending on the configuration.

    Both variants use the same prompt from memory_extractor_v5.py -
    the only difference is which model answers it.
    """
    if MEMORY_EXTRACTOR_BACKEND == "openrouter":
        from memory_extractor_cloud import CloudMemoryExtractor
        return CloudMemoryExtractor(
            model=MEMORY_EXTRACTOR_CLOUD_MODEL,
            api_key=OPENROUTER_API_KEY,
        )

    if MEMORY_EXTRACTOR_BACKEND == "ollama":
        from memory_extractor_v5 import MemoryExtractor
        # num_thread has to be passed through explicitly, otherwise the
        # extractor runs on Ollama's default instead of the core count
        # configured in config.py.
        return MemoryExtractor(
            model=MEMORY_EXTRACTOR_MODEL,
            host=OLLAMA_HOST,
            num_thread=OLLAMA_NUM_THREAD,
        )

    return None


memory = init_memory()
chroma_collection = init_chroma_collection(memory)
memory_filter = init_memory_filter()
memory_extractor = init_memory_extractor()


# ==========================================================
# Helper functions
# ==========================================================

def get_all_memories_direct(user_id):
    """Replacement for memory.get_all() - straight through Chroma, because
    mem0's own get_all() has turned out to be unreliable (it caps results
    at ~20, regardless of the limit value passed in).

    The user_id filtering is now done by Chroma itself (where=...), instead
    of loading the entire collection and sorting it out in Python.
    """
    raw = chroma_collection.get(
        where={"user_id": user_id},
        include=["metadatas", "documents"],
    )
    ids = raw.get("ids") or []
    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents") or []

    items = []
    for mem_id, md, doc in zip(ids, metadatas, documents):
        md = md or {}
        items.append({
            "id": mem_id,
            "memory": md.get("data") or doc or "",
            "metadata": {"type": md.get("type", "other")},
            "created_at": md.get("created_at"),
        })
    return items


def format_memory_with_date(mem_dict):
    """Formats a mem0 memory as 'DD.MM.YYYY: text', so that Alfred also
    knows from the context when a piece of information was learned - and
    so that it is easier to spot later in the DB what may be outdated."""
    if isinstance(mem_dict, dict):
        txt = mem_dict.get("memory") or mem_dict.get("text")
        created = mem_dict.get("created_at")
    else:
        txt = getattr(mem_dict, "memory", None) or getattr(mem_dict, "text", None)
        created = getattr(mem_dict, "created_at", None)

    if not txt:
        return None

    date_str = None
    if created:
        try:
            date_str = datetime.fromisoformat(
                str(created).replace("Z", "+00:00")
            ).strftime("%d.%m.%Y")
        except Exception:
            date_str = None

    return f"{date_str}: {txt}" if date_str else txt


def normalize_for_dedup(text):
    """Comparison form for the duplicate check.

    Capitalisation, repeated whitespace and sentence-final punctuation
    are normalised, so that 'Anna likes pizza.' and 'Anna likes pizza'
    are recognised as the same fact instead of being stored twice.
    """
    t = unicodedata.normalize("NFKC", text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.rstrip(".!?;:, ")
    return t


@st.cache_data(ttl=60)
def get_available_emojis():
    """Read the emoji file names once a minute instead of on every message."""
    if not os.path.exists(EMOJI_DIR):
        return []
    return sorted(
        f[:-4] for f in os.listdir(EMOJI_DIR) if f.lower().endswith(".png")
    )


def render_message_with_emojis(text):
    """Renders Alfred's text and lists the active emojis underneath it."""
    pattern = r"\[EMOJI:([a-zA-Z0-9_\-]+)\]"
    found_emojis = re.findall(pattern, text)
    clean_text = re.sub(pattern, "", text).strip()

    st.write(clean_text)

    if found_emojis:
        with st.container():
            for emoji_name in found_emojis:
                emoji_path = os.path.join(EMOJI_DIR, f"{emoji_name}.png")
                if os.path.exists(emoji_path):
                    st.image(emoji_path, width=45)


def load_memory_candidates(user_id):
    """All stored memories of the user, newest first."""
    all_memories = get_all_memories_direct(user_id)
    all_memories = sorted(
        all_memories,
        key=lambda m: m.get("created_at") or "",
        reverse=True,
    )

    texts = []
    seen = set()

    for m in all_memories:
        display = format_memory_with_date(m)
        key = m.get("memory")
        if display and key and key not in seen:
            seen.add(key)
            texts.append(display)

    return texts


def apply_memory_filter(user_input, memory_texts):
    """Picks the relevant memories - or returns all of them.

    Returns: (memories, note_text_for_the_ui)

    Principle: when in doubt, send too much rather than too little. One
    superfluous memory in the context costs fractions of a cent; a
    missing one costs the right answer.
    """
    if not memory_texts:
        return [], "no memories available"

    if memory_filter is None:
        return memory_texts, "no filter active - all memories in the context"

    if len(memory_texts) < MEMORY_FILTER_THRESHOLD:
        return memory_texts, (
            f"filter skipped ({len(memory_texts)} < "
            f"threshold {MEMORY_FILTER_THRESHOLD}) - all memories in the context"
        )

    try:
        selected = memory_filter.select_memories(
            user_prompt=user_input,
            memories=memory_texts,
        )
    except Exception as e:
        return memory_texts, f"filter failed ({e}) - using all memories"

    # Only let through texts that really come from the candidates.
    selected = [m for m in selected if m in memory_texts]

    if not selected:
        # The filter marked nothing. We then send everything along
        # anyway: in practice "the filter found nothing" was almost
        # always a filter error, not a real result.
        return memory_texts, "filter marked nothing - sending all memories to be safe"

    return selected, f"{len(selected)} of {len(memory_texts)} memories selected"


def build_system_prompt(user_name, memories):
    """Builds Alfred's system prompt including the memory block."""

    if memories:
        memory_block = (
            "\n\nYOUR MEMORY ABOUT THE USER\n"
            "Format: 'DD.MM.YYYY: fact' - the date says when you learned "
            "this information, not when it applies. In case of contradictory "
            "statements the more recent one is the more current one.\n"
            "This is a complete list, not a preselection. Use only what fits "
            "the current question - do not list facts unprompted "
            "that nobody asked about.\n\n"
            + "\n".join(memories)
        )
    else:
        memory_block = (
            "\n\nYOUR MEMORY ABOUT THE USER\n"
            "No stored memories yet. Do not pretend to know "
            "anything about the user - rather ask."
        )

    # Location reference only if one is stored in the configuration. Without
    # an entry Alfred leaves it out instead of inventing a location.
    ort_block = (
        f"\n\nYour physical location is {USER_LOCATION}."
        if USER_LOCATION else ""
    )

    emoji_list_str = ", ".join(get_available_emojis()) or "No emojis available"

    return f"""You are Alfred, a helpful, polite and intelligent AI assistant with a long-term memory.
The name of the user you are talking to is {user_name}. Always answer in {ASSISTANT_LANGUAGE_NAME}.{memory_block}{ort_block}

You have access to a real web search for current information (e.g. weather, news, current events) - you decide for yourself when a search is necessary. If you do search, answer directly with the information you found - do not announce the search first ("One moment, I'm searching...") and do not make the user wait, because the result is already part of the same answer.

IMPORTANT: For questions about weather, news, current events, prices or other time-critical/current information you MUST use the web search - NEVER guess or invent plausible values (temperatures, percentages, dates etc.) from your own knowledge. If you are unsure whether your information is current enough, use the web search. A wrong, invented answer is worse than one that takes a little longer.

IMPORTANT FOR YOUR EMOJIS:
You own exclusive custom emojis. When you want to express a feeling or underline a topic, write EXACTLY the format '[EMOJI:name]' in the middle of your answer text. NEVER use the .png extension in the text!

Use the emojis strictly according to the following rules:
* [EMOJI:robot_butler] - Your default picture. Shows you as friendly, nice and helpful. Use it often as a greeting!
* [EMOJI:robot_happy] - When you are simply happy or are delivering positive news.
* [EMOJI:robot_wink] - When you think something is cool or want to give the user an approving wink.
* [EMOJI:robot_coffee] - When the user is tired, asks for coffee or the two of you have finished an important task together.
* [EMOJI:robot_neutral] - When you answer matter-of-factly and are in neither a particularly positive nor a negative mood.
* [EMOJI:robot_confused] - When a question is unclear or you find something odd.
* [EMOJI:robot_surprised] - When you are surprised by something the user said.
* [EMOJI:robot_laugh] - When you find something funny, make a humorous remark or tell a joke.
* [EMOJI:robot_music] - Whenever it is about songs, music theory, instruments, listening to music or composing.
* [EMOJI:robot_germany] - When the German national team, football or typically German topics are being discussed.
* [EMOJI:robot_bored] - When a task takes very long or you are jokingly bored.
* [EMOJI:robot_sad] - When you or the user are sad.
* [EMOJI:robot_angry] - When you are dissatisfied or annoyed (use this rarely, since you are very nice).

WEATHER EMOJIS (use these for weather reports or fitting mentions):
* [EMOJI:robot_sunny] - For nice weather, sunshine or heat.
* [EMOJI:robot_rain] - For rain, showers or cold, wet weather.
* [EMOJI:robot_snow] - For snow, winter, frost or severe cold.
* [EMOJI:robot_storm] - For thunderstorms, lightning and thunder (especially for current severe weather at the user's location).
* [EMOJI:robot_windy] - For strong wind, storms or windy forecasts.

Allowed emoji names: {emoji_list_str}
Write the emojis exactly as defined above in square brackets!
Use 1 to a maximum of 2 emojis per answer and never more!"""


def ask_answer_model(system_prompt, history_messages):
    """Calls the answer model. Returns (answer_text, usage_dict)."""
    data = chat_completion(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
        ],
        api_key=OPENROUTER_API_KEY,
        tools=[
            {
                "type": "openrouter:web_search",
                "parameters": {
                    "max_results": ANSWER_WEB_SEARCH_MAX_RESULTS,
                    "search_context_size": ANSWER_WEB_SEARCH_CONTEXT_SIZE,
                },
            }
        ],
        timeout=ANSWER_TIMEOUT,
        retries=2,
    )
    return extract_text(data), extract_usage(data)


def save_new_memories(user_input, user_id, user_name=None):
    """Extracts facts from the user input and stores the new ones.

    Returns a status text for the UI.
    """
    if memory_extractor is None:
        return "Extraction disabled (MEMORY_EXTRACTOR_BACKEND = 'off')."

    # The name goes with the call: one cached extractor serves every
    # user of this installation, so it must not remember a name.
    extracted = memory_extractor.extract(user_input, user_name=user_name)

    if not extracted:
        return "No new memory detected."

    existing_texts = {
        normalize_for_dedup(e.get("memory", ""))
        for e in get_all_memories_direct(user_id)
    }

    saved_count = 0
    skipped_count = 0

    for mem_item in extracted:
        text = mem_item["text"]
        norm = normalize_for_dedup(text)

        if norm in existing_texts:
            skipped_count += 1
            continue

        memory.add(
            text,
            user_id=user_id,
            infer=False,
            metadata={"type": mem_item["type"]},
        )
        existing_texts.add(norm)
        saved_count += 1

    if saved_count:
        extra = f", {skipped_count} duplicate(s) skipped." if skipped_count else "."
        return f"{saved_count} new memory/memories stored{extra}"

    return f"All {skipped_count} detected memories were already stored."


# ==========================================================
# UI
# ==========================================================

st.title("🤖 Alfred 2.0 – Your personal assistant")

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.user_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Configuration")
    st.caption(f"Running `alfred_ollama.py`")

    # Where each stage happens. Both chat variants listen on the same
    # port and look identical, and the three stages can be switched
    # between local and cloud independently - so without this you cannot
    # tell from the screen what you actually started.
    HAUS, WOLKE = "\N{HOUSE BUILDING}", "\N{CLOUD}"

    def stufe(symbol_lokal, name, wert):
        st.caption(f"{HAUS if symbol_lokal else WOLKE} **{name}** · {wert}")

    stufe(False, "Answer", f"`{ANSWER_MODEL}`")

    if MEMORY_FILTER_MODE == "off":
        stufe(True, "Filter", "off – all memories go into the context")
    elif MEMORY_FILTER_MODE == "openrouter":
        stufe(False, "Filter", f"`{MEMORY_FILTER_MODEL}` (from {MEMORY_FILTER_THRESHOLD} facts on)")
    else:
        stufe(True, "Filter", f"Ollama `{MEMORY_MANAGER_MODEL}` (from {MEMORY_FILTER_THRESHOLD} facts on)")

    if MEMORY_EXTRACTOR_BACKEND == "openrouter":
        stufe(False, "Extraction", f"`{MEMORY_EXTRACTOR_CLOUD_MODEL}`")
    elif MEMORY_EXTRACTOR_BACKEND == "ollama":
        stufe(True, "Extraction", f"Ollama `{MEMORY_EXTRACTOR_MODEL}`")
    else:
        stufe(True, "Extraction", "off – nothing new is stored")

    # getattr, so this still works against a config.py that predates
    # the setting - the sidebar is not worth a crash.
    _embed = getattr(config, "OLLAMA_EMBED_MODEL", "nomic-embed-text")
    stufe(True, "Storage", f"mem0 + Ollama `{_embed}`")

    st.caption(
        f"{HAUS} on your machine · {WOLKE} through OpenRouter. The answer "
        "is always written in the cloud - there is no local option for it."
    )

    if not OPENROUTER_API_KEY:
        st.error(
            "OPENROUTER_API_KEY is not set. Without a key Alfred "
            "cannot answer – see the comment in config.py."
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_message_with_emojis(msg["content"])
        else:
            st.write(msg["content"])

# --- Simple name sign-in, directly above the chat input ---
# Determines the user_id for mem0 (always lowercase, so that spelling
# variants like "Anna"/"anna"/"ANNA" are not accidentally treated as
# different users).
if not st.session_state.user_id:
    name_input = st.text_input(
        "What is your name?", value="", autocomplete="off",
        placeholder=f"e.g. {USER_NAME}" if USER_NAME else "e.g. Anna",
    )
    if name_input.strip():
        st.session_state.user_id = name_input.strip().lower()
        st.session_state.user_name = name_input.strip()
        st.rerun()
    st.stop()

user_id = st.session_state.user_id
user_name = st.session_state.user_name

st.caption(f"Signed in as: {user_name}")


if user_input := st.chat_input("Write Alfred a message..."):

    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    timings = {}

    # ------------------------------------------------------
    # 1. Load memories
    # ------------------------------------------------------
    t0 = time.perf_counter()
    try:
        memory_texts = load_memory_candidates(user_id)
        load_error = None
    except Exception as e:
        memory_texts = []
        load_error = str(e)
    timings["Loading memories"] = time.perf_counter() - t0

    if load_error:
        st.caption(f"⚠️ Memory retrieval failed: {load_error}")

    # ------------------------------------------------------
    # 2. Filter optionally
    # ------------------------------------------------------
    t0 = time.perf_counter()
    relevant_memories, filter_note = apply_memory_filter(user_input, memory_texts)
    timings["Memory filter"] = time.perf_counter() - t0

    st.caption(f"🧠 {filter_note}")

    # ------------------------------------------------------
    # 3. Fetch the answer
    # ------------------------------------------------------
    system_prompt = build_system_prompt(user_name, relevant_memories)

    # Short-term memory: send along the last N messages of this session
    # (including the question just asked, which is already in
    # st.session_state.messages above).
    history_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-MAX_CHAT_HISTORY_MESSAGES:]
    ]

    answer_usage = {}
    answer_failed = False

    with st.chat_message("assistant"):
        with st.spinner("Alfred is thinking..."):
            t0 = time.perf_counter()
            try:
                answer, answer_usage = ask_answer_model(system_prompt, history_messages)
            except OpenRouterError as e:
                answer = f"⚠️ **Alfred could not answer:** {e}"
                answer_failed = True
            except Exception as e:
                answer = f"⚠️ **Unexpected error:** {e}"
                answer_failed = True
            timings["Answer model"] = time.perf_counter() - t0

        render_message_with_emojis(answer)

        search_count = answer_usage.get("web_search_requests", 0)
        st.caption(
            f"🌐 Searched the web {search_count}x for this answer."
            if search_count else
            "🌐 No web search for this answer (based on model knowledge)."
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # ------------------------------------------------------
    # 4. Store new memories (after the answer)
    # ------------------------------------------------------
    # At this point the answer is already on the screen - so the user
    # is no longer waiting for the storage logic.
    if not answer_failed:
        t0 = time.perf_counter()
        try:
            save_note = save_new_memories(user_input, user_id, user_name)
        except Exception as e:
            save_note = f"⚠️ Memory error: {e}"
        timings["Memory extraction"] = time.perf_counter() - t0
        st.caption(f"💾 {save_note}")

    # ------------------------------------------------------
    # 5. Diagnostics
    # ------------------------------------------------------
    if SHOW_TIMINGS:
        parts = [f"{name}: {sec:.2f}s" for name, sec in timings.items() if sec >= 0.01]
        line = "⏱️ " + " | ".join(parts) + f" | total: {sum(timings.values()):.2f}s"

        prompt_tokens = answer_usage.get("prompt_tokens")

        # The real price of a message is all three calls, not just the
        # answer. Showing only the answer made the cloud route look
        # cheaper than it is, and hid what the local route actually
        # saves - a stage running on Ollama simply contributes nothing
        # here, because it costs nothing.
        posten = [("answer", answer_usage.get("cost"))]
        for name, komponente in (("filter", memory_filter),
                                 ("extraction", memory_extractor)):
            nutzung = getattr(komponente, "last_usage", None) if komponente else None
            if nutzung:
                posten.append((name, nutzung.get("cost")))

        bekannt = [(n, float(c)) for n, c in posten if c is not None]
        gesamt = sum(c for _, c in bekannt)

        if prompt_tokens:
            line += f"\n\n📊 {prompt_tokens} input tokens to the answer model"
        if bekannt:
            aufschluesselung = ", ".join(f"{n} ${c:.5f}" for n, c in bekannt)
            line += f" | this message cost ${gesamt:.5f} ({aufschluesselung})"

        st.caption(line)
