"""
db_cleanup.py

Interactive cleanup script for the Alfred memory database (mem0).

Two modes:
  - ENTER (empty input) -> complete wipe of all of the user's memories
  - number (days)       -> delete only memories older than N days

Always asks for an explicit confirmation before actually deleting
anything.

Run it in the Alfred directory: python db_cleanup.py
"""

from datetime import datetime, timedelta, timezone
import os

os.environ.setdefault("MEM0_TELEMETRY", "False")  # see memory_import.py

from mem0 import Memory
from config_guard import build_mem0_memory, require_config
require_config()   # readable message if config.py is missing

import config
from config import MEM0_CONFIG, USER_ID



# Read with getattr so an older config.py without this setting keeps
# working - it then uses the name the project has always used.
CHROMA_COLLECTION = getattr(config, "CHROMA_COLLECTION", "alfred_memory")
def get_chroma_collection(memory):
    """Accesses the Chroma collection that mem0 has already created
    INTERNALLY while building the Memory instance, instead of creating a
    new chromadb.PersistentClient of its own - a second, independent
    client for the same path would crash Chroma with 'An instance of
    Chroma already exists ... with different settings'."""
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
        "Could not read the Chroma collection from mem0's internal vector_store."
    )


def get_all_memories(memory, user_id):
    # Directly through Chroma instead of mem0's Memory.get_all() - the
    # latter turned out to be unreliable (it caps results, regardless of
    # the limit value passed in). The actual deleting still goes through
    # mem0 (memory.delete(memory_id=...) is a targeted single-ID
    # operation, unaffected by the listing bug).
    collection = get_chroma_collection(memory)

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
            "created_at": md.get("created_at"),
        })
    return items


def parse_created_at(raw):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def confirm(frage):
    antwort = input(f"{frage} (yes/no): ").strip().lower()
    return antwort in ("j", "ja", "y", "yes")


def wipe_all(memory, items):
    print(f"\n{len(items)} memory/memories will be deleted IRREVERSIBLY.")
    if not confirm("Are you sure you REALLY want to delete EVERYTHING?"):
        print("Cancelled.")
        return

    deleted = 0
    for item in items:
        mem_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        if not mem_id:
            continue
        try:
            memory.delete(memory_id=mem_id)
            deleted += 1
        except Exception as e:
            print(f"  ✗ could not be deleted ({mem_id}): {e}")

    print(f"\n✓ {deleted} of {len(items)} memory/memories deleted.")


def wipe_older_than(memory, items, days):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    to_delete = []
    for item in items:
        created = parse_created_at(item.get("created_at") if isinstance(item, dict) else getattr(item, "created_at", None))
        if created and created < cutoff:
            to_delete.append(item)

    if not to_delete:
        print(f"\nNo memories older than {days} day(s) found. Nothing to do.")
        return

    print(f"\nThe following {len(to_delete)} memory/memories are older than {days} day(s) and would be deleted:")
    for item in to_delete:
        txt = item.get("memory") if isinstance(item, dict) else getattr(item, "memory", "?")
        created = item.get("created_at") if isinstance(item, dict) else getattr(item, "created_at", "?")
        print(f"  - [{created}] {txt}")

    if not confirm("\nDelete these memories now?"):
        print("Cancelled.")
        return

    deleted = 0
    for item in to_delete:
        mem_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        if not mem_id:
            continue
        try:
            memory.delete(memory_id=mem_id)
            deleted += 1
        except Exception as e:
            print(f"  ✗ could not be deleted ({mem_id}): {e}")

    print(f"\n✓ {deleted} of {len(to_delete)} memory/memories deleted.")


def main():
    memory = build_mem0_memory(MEM0_CONFIG)

    user_id = input(f"Which user_id should be cleaned up? [{USER_ID}]: ").strip()
    if not user_id:
        user_id = USER_ID

    items = get_all_memories(memory, user_id)
    print(f"Currently stored memories for '{user_id}': {len(items)}")

    eingabe = input(
        "\nENTER = delete everything (complete wipe)\n"
        "number = delete only memories older than X days\n"
        "Input: "
    ).strip()

    if eingabe == "":
        wipe_all(memory, items)
    else:
        try:
            days = int(eingabe)
            if days < 0:
                raise ValueError
        except ValueError:
            print("Invalid input - please enter ENTER or a positive whole number.")
            return
        wipe_older_than(memory, items, days)

    print("\nStatus afterwards:")
    after = get_all_memories(memory, user_id)
    print(f"  {len(after)} memory/memories remain for '{user_id}'.")


if __name__ == "__main__":
    main()
