"""
alfred.py

Alfred 2.0 - the chat client. START HERE.

A personal assistant with a long-term memory: it remembers facts about
you across sessions, and you can inspect, consolidate and delete that
memory with the tools next to it.

WHAT THIS VARIANT NEEDS
-----------------------
An OpenRouter API key. Nothing else - no local AI installation, no
Ollama, no GPU. Everything that needs a model goes to OpenRouter, and
the memory is stored directly in a local Chroma database.

There is a second variant, alfred_ollama.py, which stores through mem0
and therefore needs a local Ollama. Both share the same database, the
same config.py and all the tools around them - see the comparison in
alfred_ollama.py. If you are unsure: use this one.

HOW IT WORKS, IN ONE PASS
-------------------------
1. Your message arrives.
2. The memories relevant to it are selected (MEMORY_FILTER_MODE). The
   filter model is only ever shown numbered memories and only ever
   returns ID numbers - never text. It therefore cannot reword or invent
   a memory, and it works the same in any language.
3. The selected memories go into the system prompt of the answer model,
   which writes the reply.
4. AFTER the answer is on screen, new facts are extracted from the
   exchange and stored. You never wait on the storage logic
   (MEMORY_EXTRACTION_DEFERRED).

Below every answer you can see how long each stage took and what the
request actually cost according to OpenRouter (SHOW_TIMINGS) - so tuning
decisions can be made on numbers rather than on guesses.

WHY NO mem0 AT RUNTIME
----------------------
mem0's Memory.add() wants an embedding vector for EVERY stored memory,
even with infer=False, and producing one meant running a local embedding
model. This client talks to Chroma directly instead, the same way
memory_viewer.py already does for reading. That is what makes it run
without any local AI installation.

The vectors it writes are zero vectors: nothing here searches by
similarity, only the dimension has to match the collection.

Start:  streamlit run alfred.py          (or start_alfred.bat)
"""

import os
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone

import chromadb
import streamlit as st

from config_guard import require_config
require_config()   # readable message if config.py is missing

import config
from config import (
    ANSWER_MODEL,
    ANSWER_TIMEOUT,
    ANSWER_WEB_SEARCH_CONTEXT_SIZE,
    ANSWER_WEB_SEARCH_MAX_RESULTS,
    CHROMA_DB_DIR,
    EMOJI_DIR,
    MAX_CHAT_HISTORY_MESSAGES,
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
def init_chroma_collection():
    """Direct Chroma access, without instantiating mem0's Memory class.

    alfred_ollama.py goes through mem0 (Memory.from_config), because
    mem0 brings its own search and extraction logic. This client does not
    need that: reading runs through collection.get() (see
    get_all_memories_direct), writing through collection.add() (see
    save_memory_direct) - both without any embedding computation by an
    external model. That is what leaves this client with no runtime Ollama
    dependency at all, not even indirectly via mem0.
    """
    collection_name = CHROMA_COLLECTION
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(collection_name)


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


chroma_collection = init_chroma_collection()
memory_filter = init_memory_filter()
memory_extractor = init_memory_extractor()


# ==========================================================
# Helper functions
# ==========================================================

def get_embedding_dimension(collection):
    """Determines the vector dimension from an existing memory instead of
    guessing it (nomic-embed-text returns 768, but this code here does not
    rely on that detail - if the collection was filled with a different
    embedder, it adapts automatically).

    Falls back to 768 if the collection is still completely empty
    (a fresh start without a chroma_db brought along) - in that case the
    value is meaningless anyway, because nothing here ever uses these
    vectors for a similarity search.
    """
    probe = collection.get(limit=1, include=["embeddings"])
    embeddings = probe.get("embeddings")
    if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
        return len(embeddings[0])
    return 768


def save_memory_direct(collection, user_id, text, mem_type):
    """Stores a memory directly in Chroma, without mem0 and without a
    real embedding computation.

    The vector is deliberately a zero vector: this client reads memories
    exclusively through collection.get() with a metadata filter (see
    get_all_memories_direct), never through a similarity search - so the
    content of the vector is irrelevant to it. The dimension is
    nevertheless matched to the existing data, so that Chroma does not
    abort because of a dimension conflict.

    CAUTION if you ever point something at this database that DOES search
    by similarity: memories stored this way are practically impossible for
    it to find, because a zero vector has no meaningful proximity to
    anything. Nothing shipped here does that - the memory filter works on
    numbered text, not on vectors - but memory_import.py can recompute
    real embeddings via mem0/Ollama if you need them.
    """
    dim = get_embedding_dimension(collection)
    now = datetime.now(timezone.utc).isoformat()
    mem_id = str(uuid.uuid4())

    collection.add(
        ids=[mem_id],
        documents=[text],
        embeddings=[[0.0] * dim],
        metadatas=[{
            "user_id": user_id,
            "type": mem_type,
            "data": text,
            "created_at": now,
        }],
    )
    return mem_id


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
Write the emojis exactly as defined above in square brackets!"""


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

        save_memory_direct(chroma_collection, user_id, text, mem_item["type"])
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
    st.caption(f"Running `alfred.py`")

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

    stufe(True, "Storage", "Chroma directly, no model involved")

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

    # With filtering active: make it traceable WHAT was selected.
    # Seeing only the count is not enough to judge the quality of a
    # filter - only the comparison "selected vs. discarded" shows
    # whether the model caught the right facts.
    if SHOW_TIMINGS and memory_texts and len(relevant_memories) < len(memory_texts):
        verworfen = [m for m in memory_texts if m not in relevant_memories]

        with st.expander(
            f"🔎 Filter details: {len(relevant_memories)} selected, "
            f"{len(verworfen)} discarded"
        ):
            st.markdown("**Selected (goes to the answer model):**")
            for m in relevant_memories:
                st.markdown(f"- {m}")

            st.markdown("**Discarded:**")
            for m in verworfen:
                st.markdown(f"- {m}")

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
