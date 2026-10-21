"""
memory_export.py

Alfred 2.0 - database export

Counterpart to memory_import.py - it writes exactly the format that
memory_import.py's main() reads back in: {"exported_at": ...,
"memories": [{"type","text","created_at"}, ...]}. Each entry
additionally carries "id" and "user_id" - memory_import.py does not read
those, it simply ignores unknown fields, but they help when reviewing or
correcting the file by hand.

Usable on its own as a plain backup, independent of any consolidation.
Deliberately WITHOUT embeddings in the export - nothing here uses them
for a similarity search anyway, and memory_import.py computes new ones
via mem0/Ollama when reading the data back in.

Usage:

    python memory_export.py                  Exports USER_ID from config.py
    python memory_export.py --user anna      Exports one specific user
    python memory_export.py --all             Exports ALL users in the DB
                                               (when reading the data back in,
                                               memory_import.py asks for only
                                               ONE target user - so use --all
                                               only for reviewing or as a full
                                               backup, do not re-import it
                                               directly like this)

Writes to exports/chroma_export_<timestamp>.json
"""

import argparse
import json
import os
from datetime import datetime

import chromadb

from config_guard import require_config
require_config()   # readable message if config.py is missing

import config
from config import CHROMA_DB_DIR, USER_ID


# Read with getattr so an older config.py without this setting keeps
# working - it then uses the name the project has always used.
CHROMA_COLLECTION = getattr(config, "CHROMA_COLLECTION", "alfred_memory")
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")


def get_collection():
    collection_name = CHROMA_COLLECTION
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(collection_name)


def export_all(collection, user_filter=None):
    """user_filter=None -> all users. Otherwise only the given user_id."""
    where = {"user_id": user_filter} if user_filter else None
    raw = collection.get(where=where, include=["metadatas", "documents"])

    ids = raw.get("ids") or []
    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents") or []

    records = []
    for mem_id, md, doc in zip(ids, metadatas, documents):
        md = md or {}
        records.append({
            "id": mem_id,
            "user_id": md.get("user_id", ""),
            "type": md.get("type", "other"),
            "text": md.get("data") or doc or "",
            "created_at": md.get("created_at"),
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Alfred Memory Export")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--user", help="Export only this user_id")
    group.add_argument("--all", action="store_true", help="Export all users")
    args = parser.parse_args()

    user_filter = None if args.all else (args.user or USER_ID)

    collection = get_collection()
    records = export_all(collection, user_filter)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    zeitstempel = datetime.now().strftime("%Y%m%d_%H%M%S")
    ziel = os.path.join(EXPORT_DIR, f"chroma_export_{zeitstempel}.json")

    # Top-level keys named exactly the way memory_import.py reads them
    # (exported_at, memories) - see its main(). anzahl/user_filter are
    # additional metadata for human eyes, ignored by memory_import.py.
    payload = {
        "exported_at": datetime.now().isoformat(),
        "user_filter": user_filter or "(all)",
        "anzahl": len(records),
        "memories": records,
    }

    with open(ziel, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"{len(records)} memory/memories exported to:\n  {ziel}")
    if not records:
        print(
            "Note: 0 memories found - are user_id/CHROMA_DB_DIR correct? "
            f"(filter: {user_filter or 'all'})"
        )


if __name__ == "__main__":
    main()
