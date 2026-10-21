"""
memory_import.py

Counterpart to memory_export.py: reads a JSON file in the known export
format (exported_at, memories: [{type, text, created_at}]) and rebuilds
the memories from it in mem0. Intended for this workflow: export ->
correct mistakes in the JSON file by hand -> clean up the DB (e.g. with
db_cleanup.py) -> read it back in with this script.

Deliberately keeps using mem0/Ollama for memory.add() - that produces
real embeddings rather than the zero vectors alfred.py writes, so
imported memories are also findable by a similarity search if you ever
point one at this database. READING (duplicate check, final status) goes
directly through Chroma instead of mem0's own get_all(), which is
unreliable with larger amounts of data (it apparently caps results at
~20, regardless of the limit value passed in - see memory_viewer.py,
alfred.py, filter_compare.py, where it is worked around for the
same reason).

Run it in the Alfred directory: python memory_import.py
"""

import json
import os
import re
import unicodedata
from pathlib import Path

# Disables mem0's built-in PostHog telemetry. MUST be set before the
# mem0 import. Known mem0 versions still occasionally fire off
# connection attempts internally anyway (see mem0 GitHub issues
# #3729/#3762) - if the error messages show up regardless, they are
# harmless (mem0 catches them itself) and do not prevent the import,
# they are merely annoying in the console.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory

from config_guard import build_mem0_memory, require_config
require_config()   # readable message if config.py is missing

import config
from config import MEM0_CONFIG, USER_NAME



# Read with getattr so an older config.py without this setting keeps
# working - it then uses the name the project has always used.
CHROMA_COLLECTION = getattr(config, "CHROMA_COLLECTION", "alfred_memory")
def normalize_for_dedup(text):
    """Comparison form for the duplicate check - identical to the one in
    alfred.py. Normalizes capitalization, repeated whitespace and
    sentence-final punctuation, so that 'the user likes pizza.' and
    'the user likes pizza' are recognized as the same thing."""
    t = unicodedata.normalize("NFKC", text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t.rstrip(".!?;:, ")


def get_chroma_collection(memory):
    """Fetches the Chroma collection from mem0's already existing Memory
    instance instead of creating a second, independent
    chromadb.PersistentClient in the same process - that would crash
    Chroma with 'An instance of Chroma already exists ... with different
    settings', since Chroma allows only one client process per path."""
    vector_store = getattr(memory, "vector_store", None)
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


def get_all_memories(collection, user_id):
    """Directly through Chroma instead of mem0's Memory.get_all() - see
    the docstring above."""
    raw = collection.get(where={"user_id": user_id}, include=["metadatas", "documents"])
    ids = raw.get("ids") or []
    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents") or []

    items = []
    for mem_id, md, doc in zip(ids, metadatas, documents):
        md = md or {}
        items.append({"id": mem_id, "memory": md.get("data") or doc or ""})
    return items


def confirm(frage):
    antwort = input(f"{frage} (yes/no): ").strip().lower()
    return antwort in ("j", "ja", "y", "yes")


def main():
    memory = build_mem0_memory(MEM0_CONFIG)
    collection = get_chroma_collection(memory)

    user_id = input(f"Import for which user? [{USER_NAME.lower()}]: ").strip().lower()
    if not user_id:
        user_id = USER_NAME.lower()

    input_path = input("JSON file to read [memory_export.json]: ").strip()
    if not input_path:
        input_path = "memory_export.json"

    path = Path(input_path)
    if not path.exists():
        print(f"\n✗ File not found: {path.resolve()}")
        return

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\n✗ Invalid JSON in {path}: {e}")
        print("  Please correct the file and try again.")
        return

    entries = data.get("memories")
    if not isinstance(entries, list):
        print("\n✗ File contains no 'memories' array in the expected format.")
        return

    # Load existing memories to avoid exact duplicates - in case the DB
    # was not emptied beforehand or the script accidentally runs twice.
    existing = get_all_memories(collection, user_id)
    existing_texts = {normalize_for_dedup(e.get("memory", "")) for e in existing}

    valid_entries = []
    skipped_invalid = 0
    for entry in entries:
        text = (entry.get("text") or "").strip()
        if not text:
            skipped_invalid += 1
            continue
        valid_entries.append(entry)

    print(f"\n{len(entries)} entry/entries in the file, {len(valid_entries)} of them with valid text"
          + (f" ({skipped_invalid} without text skipped)." if skipped_invalid else "."))

    if not valid_entries:
        print("Nothing to import.")
        return

    print(f"Target: user '{user_id}', currently {len(existing)} existing memory/memories in the DB.")
    if not confirm(f"Import {len(valid_entries)} memory/memories now?"):
        print("Cancelled.")
        return

    imported = 0
    skipped_duplicate = 0
    failed = 0

    for entry in valid_entries:
        text = entry["text"].strip()
        norm = normalize_for_dedup(text)

        if norm in existing_texts:
            skipped_duplicate += 1
            continue

        mem_type = entry.get("type", "other")
        original_created_at = entry.get("created_at")

        try:
            # infer=False: the text is taken over raw, no renewed LLM
            # extraction by mem0. original_created_at is stored along as a
            # metadata field, since mem0 presumably sets the real
            # created_at to "now" on import rather than to the original
            # point in time.
            memory.add(
                text,
                user_id=user_id,
                infer=False,
                metadata={"type": mem_type, "original_created_at": original_created_at},
            )
            existing_texts.add(norm)
            imported += 1
        except Exception as e:
            print(f"  ✗ Error importing '{text}': {e}")
            failed += 1

    print(f"\n✓ {imported} imported, {skipped_duplicate} duplicate(s) skipped, {failed} failed.")

    after = get_all_memories(collection, user_id)
    print(f"Current status: {len(after)} memory/memories for '{user_id}'.")


if __name__ == "__main__":
    main()
