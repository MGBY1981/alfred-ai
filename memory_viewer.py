"""
memory_viewer.py

Standalone Streamlit tool for inspecting and searching the Alfred memory
database (mem0), detached from the actual chat.

Start: streamlit run memory_viewer.py
(separately from the chat, by default on a different port)
"""

from datetime import datetime, timezone
import os

os.environ.setdefault("MEM0_TELEMETRY", "False")  # see memory_import.py

import chromadb
import streamlit as st

from config_guard import require_config
require_config()   # readable message if config.py is missing

import config
from config import CHROMA_DB_DIR, USER_NAME

# Read with getattr so an older config.py without this setting keeps
# working - it then uses the name the project has always used.
CHROMA_COLLECTION = getattr(config, "CHROMA_COLLECTION", "alfred_memory")
from language_packs import get_pack
from memory_extractor_v5 import ALLOWED_TYPES

st.set_page_config(page_title="Alfred Memory Viewer", page_icon="🔍")


@st.cache_resource
def init_chroma_collection():
    """The Chroma collection, or None if it does not exist yet.

    Deliberately get_collection and NOT get_or_create_collection: creating
    the collection is the chat's job, because it is the chat that knows
    which embedding model belongs to it. On a fresh clone there is
    simply nothing here yet - that is not an error worth a traceback, so
    the caller shows a short note instead.
    """
    collection_name = CHROMA_COLLECTION
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        return client.get_collection(collection_name)
    except Exception:
        return None


def format_date(raw):
    if not raw:
        return "?"
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return str(raw)


# Words whose occurrence indicates that a sentence actually has a subject
# and a predicate (a rough, but usable heuristic). Not exhaustive - it only
# serves to find suspicious candidates more easily, not to decide
# automatically what gets deleted. Comes from the language pack, so this is
# the same table alfred_sleep.py uses. Read with getattr so an older
# config.py without the language setting still works (English then).
_VERB_HINWEISE = get_pack(getattr(config, "ASSISTANT_LANGUAGE", "en"))["verb_hints"]


def looks_like_fragment(text):
    """Rough heuristic for presumably subject-less/verb-less fragments like
    'LEGO' or 'Champions League' (see the chat history on the extractor bug
    with enumerations). Only a hint for searching, NOT a reliable
    detection - always proof-read yourself before deleting.

    With no verb list for the configured language nothing is flagged -
    an empty table would otherwise mark every short memory."""
    if not _VERB_HINWEISE:
        return False
    woerter = text.strip().rstrip(".!?").split()
    if not woerter:
        return False
    if len(woerter) > 3:
        return False
    return not any(w.lower() in _VERB_HINWEISE for w in woerter)


def get_all_memories(collection, user_id):
    # Directly via Chroma instead of via mem0's Memory.get_all() - the
    # latter has turned out to be unreliable (it apparently caps results at
    # ~20, regardless of the limit value passed in). Chroma itself reliably
    # returns everything. alfred.py, memory_import.py and filter_compare.py
    # take the same route for the same reason.
    raw = collection.get(include=["metadatas", "documents"])
    ids = raw.get("ids") or []
    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents") or []

    items = []
    for mem_id, md, doc in zip(ids, metadatas, documents):
        md = md or {}
        if md.get("user_id") != user_id:
            continue
        items.append({
            "id": mem_id,
            "memory": md.get("data") or doc or "",
            "metadata": {"type": md.get("type", "other")},
            "created_at": md.get("created_at"),
            "edited_at": md.get("edited_at"),
            "user_id": md.get("user_id", user_id),
            # Where the entry comes from. Both fields are only set by the
            # newer tools and are missing on everything that comes from the
            # normal chat - which is exactly what makes them useful as a
            # filter.
            "source": md.get("source"),
            "consolidated_at": md.get("consolidated_at"),
        })
    return items


def delete_memories(collection, ids):
    """Irrevocably deletes the given IDs from Chroma."""
    ids = [i for i in ids if i]
    if not ids:
        return 0
    collection.delete(ids=ids)
    return len(ids)


def update_memory(collection, mem_id, user_id, created_at, new_text, new_type):
    """Overwrites the text and/or the type of an existing memory.

    collection.update() leaves fields that are not passed in (here:
    embeddings) untouched - an existing real or placeholder vector is
    therefore preserved exactly as it was. created_at likewise stays
    unchanged (when the fact was originally learned); instead a separate
    edited_at is set, so that it stays traceable that and when a correction
    was made.
    """
    new_meta = {
        "user_id": user_id,
        "type": new_type,
        "data": new_text,
        "created_at": created_at,
        "edited_at": datetime.now(timezone.utc).isoformat(),
    }
    collection.update(ids=[mem_id], documents=[new_text], metadatas=[new_meta])


collection = init_chroma_collection()

st.title("🔍 Alfred Memory Viewer")
st.caption(
    "A separate inspection and clean-up tool - it has nothing to do with "
    "the actual chat. Deletions are immediate and irrevocable."
)

if collection is None:
    st.info(
        "There is no memory database yet. Start the chat "
        "(`start_alfred.bat` or `streamlit run alfred.py`) and talk "
        "to Alfred once - the database is created on the first memory. "
        "After that this viewer has something to show."
    )
    st.stop()

# --- Select the user ---
user_id_input = st.text_input("Whose memories do you want to see?", value=USER_NAME,
                               autocomplete="off").strip().lower()

if not user_id_input:
    st.info("Please enter a name.")
    st.stop()

items = get_all_memories(collection, user_id_input)

if not items:
    st.warning(f"No memories found for '{user_id_input}'.")
    st.stop()

st.write(f"**{len(items)}** memory/memories in total for **{user_id_input}**.")

if "confirm_delete_id" not in st.session_state:
    st.session_state.confirm_delete_id = None
if "editing_id" not in st.session_state:
    st.session_state.editing_id = None
if "selected_ids" not in st.session_state:
    st.session_state.selected_ids = set()
if "confirm_bulk_delete" not in st.session_state:
    st.session_state.confirm_bulk_delete = False
if "seite" not in st.session_state:
    st.session_state.seite = 1

# --- Filter bar ---
col1, col2 = st.columns([2, 1])

with col1:
    search_query = st.text_input("🔎 Search the memories (text contains...)", value="")

with col2:
    all_types = sorted({
        (item.get("metadata") or {}).get("type", "other")
        if isinstance(item, dict) else "other"
        for item in items
    })
    type_filter = st.selectbox("Type", options=["All"] + all_types)

herkunft_filter = st.selectbox(
    "Origin",
    options=["All", "Learned from the chat", "From AI dialogues", "Consolidated (alfred_sleep)"],
    help=(
        "Memories originating from AI dialogues are hearsay and carry "
        "source=dialog:<id>. Consolidated ones came into being during "
        "sleeping. Everything without either comes from a real "
        "conversation with you."
    ),
)

fragments_only = st.checkbox(
    "⚠️ Only show suspicious fragments (short, no recognizable subject/verb)",
    help=(
        "A rough heuristic, not a reliable test - it only helps to track "
        "down memories like 'LEGO' or 'Champions League' that lost their "
        "subject during extraction. Always proof-read yourself."
    ),
)

# --- Filtering ---
filtered = items

if search_query.strip():
    q = search_query.strip().lower()
    filtered = [
        item for item in filtered
        if q in (item.get("memory") or "").lower()
    ]

if type_filter != "All":
    filtered = [
        item for item in filtered
        if (item.get("metadata") or {}).get("type", "other") == type_filter
    ]

if herkunft_filter == "From AI dialogues":
    filtered = [i for i in filtered if (i.get("source") or "").startswith("dialog:")]
elif herkunft_filter == "Consolidated (alfred_sleep)":
    filtered = [i for i in filtered if i.get("consolidated_at")]
elif herkunft_filter == "Learned from the chat":
    filtered = [i for i in filtered
                if not (i.get("source") or "").startswith("dialog:")
                and not i.get("consolidated_at")]

if fragments_only:
    filtered = [
        item for item in filtered
        if looks_like_fragment(item.get("memory") or "")
    ]

# --- Sorting: newest first ---
filtered = sorted(
    filtered,
    key=lambda i: i.get("created_at") or "",
    reverse=True,
)

st.write(f"Showing **{len(filtered)}** of {len(items)} memory/memories.")

# --- Bulk deletion ---
filtered_ids = {item["id"] for item in filtered}

# Limit the selection to the currently visible hits, so that a change of
# filter does not accidentally delete invisible memories along with them.
st.session_state.selected_ids &= filtered_ids

col_a, col_b, col_c = st.columns([1, 1, 2])

with col_a:
    if filtered and st.button(f"Select all {len(filtered)} visible ones"):
        st.session_state.selected_ids |= filtered_ids
        st.rerun()

with col_b:
    if st.session_state.selected_ids and st.button("Clear selection"):
        st.session_state.selected_ids.clear()
        st.session_state.confirm_bulk_delete = False
        st.rerun()

anzahl_ausgewaehlt = len(st.session_state.selected_ids)

with col_c:
    if anzahl_ausgewaehlt:
        if not st.session_state.confirm_bulk_delete:
            if st.button(f"🗑 Delete {anzahl_ausgewaehlt} selected"):
                st.session_state.confirm_bulk_delete = True
                st.rerun()
        else:
            st.warning(
                f"Delete {anzahl_ausgewaehlt} memory/memories IRREVOCABLY?"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, delete for good", key="bulk_confirm_yes"):
                    n = delete_memories(collection, st.session_state.selected_ids)
                    st.session_state.selected_ids.clear()
                    st.session_state.confirm_bulk_delete = False
                    st.success(f"{n} memory/memories deleted.")
                    st.rerun()
            with c2:
                if st.button("✖ Cancel", key="bulk_confirm_no"):
                    st.session_state.confirm_bulk_delete = False
                    st.rerun()

# --- Delete confirmation RIGHT AT THE TOP ---------------------------------
# It used to sit at the bottom of the respective card. With a handful of
# memories that was fine; with several hundred it was not any more: after
# the click on the wastebasket, Streamlit reloads the page and jumps to the
# top while doing so - the confirmation prompt then appeared hundreds of
# cards further down and was simply invisible. It looked as if the button
# did nothing.
# That is why it now sits at a fixed place that you always see.
if st.session_state.confirm_delete_id:
    zu_loeschen = next(
        (i for i in items if i["id"] == st.session_state.confirm_delete_id), None
    )
    if zu_loeschen is None:
        st.session_state.confirm_delete_id = None
    else:
        st.warning(f"Really delete?\n\n**{zu_loeschen.get('memory', '')}**")
        b1, b2, _ = st.columns([1, 1, 3])
        with b1:
            if st.button("✅ Yes, delete", key="single_confirm_yes", type="primary"):
                delete_memories(collection, [st.session_state.confirm_delete_id])
                st.session_state.selected_ids.discard(st.session_state.confirm_delete_id)
                st.session_state.confirm_delete_id = None
                st.rerun()
        with b2:
            if st.button("✖ Cancel", key="single_confirm_no"):
                st.session_state.confirm_delete_id = None
                st.rerun()

st.divider()

if not filtered:
    st.info("No memories match this filter.")

# --- Pagination -----------------------------------------------------------
# The second reason why it used to hang: every card creates about half a
# dozen Streamlit elements. With 640 memories that is more than four
# thousand, which are completely rebuilt on EVERY click. With a page size
# of 50 that stays in the double digits and reacts instantly.
if filtered:
    p1, p2, p3 = st.columns([1, 2, 2])

    with p1:
        pro_seite = st.selectbox(
            "per page", options=[25, 50, 100, "all"], index=1,
            help="“all” is very slow when there are many memories.",
        )

    if pro_seite == "all":
        seiten, seite = 1, 1
        sichtbar = filtered
    else:
        seiten = max(1, (len(filtered) + pro_seite - 1) // pro_seite)
        st.session_state.seite = min(max(1, st.session_state.seite), seiten)

        with p2:
            seite = st.number_input(
                "Page", min_value=1, max_value=seiten,
                value=st.session_state.seite, step=1, key="seite_eingabe",
            )
            st.session_state.seite = seite

        start = (seite - 1) * pro_seite
        sichtbar = filtered[start:start + pro_seite]

    with p3:
        st.caption(
            f"Page {seite} of {seiten} · showing {len(sichtbar)} memory/memories"
        )
else:
    sichtbar = []

for item in sichtbar:
    mem_type = (item.get("metadata") or {}).get("type", "other")
    text = item.get("memory", "")
    created = format_date(item.get("created_at"))
    mem_id = item.get("id", "?")
    verdacht = looks_like_fragment(text)

    with st.container(border=True):
        row = st.columns([0.06, 0.06, 0.7, 0.09, 0.09])

        with row[0]:
            ausgewaehlt = st.checkbox(
                "select", value=mem_id in st.session_state.selected_ids,
                key=f"sel_{mem_id}", label_visibility="collapsed",
            )
            if ausgewaehlt:
                st.session_state.selected_ids.add(mem_id)
            else:
                st.session_state.selected_ids.discard(mem_id)

        with row[1]:
            if verdacht:
                st.markdown("⚠️")

        with row[2]:
            st.markdown(f"**[{mem_type}]** {text}")
            caption = f"Saved: {created} · ID: {mem_id}"
            if item.get("edited_at"):
                caption += f" · Edited: {format_date(item['edited_at'])}"
            st.caption(caption)
            if verdacht:
                st.caption("⚠️ Looks like a fragment without a recognizable subject/verb.")

        with row[3]:
            if st.button("✏️", key=f"edit_{mem_id}", help="Edit"):
                st.session_state.editing_id = mem_id
                st.session_state.confirm_delete_id = None
                st.rerun()

        with row[4]:
            if st.button("🗑️", key=f"del_{mem_id}", help="Delete"):
                st.session_state.confirm_delete_id = mem_id
                st.session_state.editing_id = None
                st.rerun()

        # --- Edit form, full card width ---
        if st.session_state.editing_id == mem_id:
            neuer_text = st.text_area(
                "Text", value=text, key=f"edittext_{mem_id}", height=100,
            )

            type_optionen = sorted(ALLOWED_TYPES)
            aktueller_index = type_optionen.index(mem_type) if mem_type in type_optionen else 0
            neuer_typ = st.selectbox(
                "Type", options=type_optionen, index=aktueller_index, key=f"edittype_{mem_id}",
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Save", key=f"save_{mem_id}"):
                    if neuer_text.strip():
                        update_memory(
                            collection, mem_id,
                            user_id=item.get("user_id", user_id_input),
                            created_at=item.get("created_at"),
                            new_text=neuer_text.strip(),
                            new_type=neuer_typ,
                        )
                        st.session_state.editing_id = None
                        st.success("Saved.")
                        st.rerun()
                    else:
                        st.error("The text must not be empty - to remove it, delete it instead.")
            with c2:
                if st.button("✖ Cancel", key=f"canceledit_{mem_id}"):
                    st.session_state.editing_id = None
                    st.rerun()
