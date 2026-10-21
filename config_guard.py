"""
config_guard.py

Startup guards: a readable message instead of a traceback when something
the program needs is not there yet.

Two cases live here - a missing config.py (require_config), and the
optional local stack of mem0 plus Ollama (build_mem0_memory).

config.py is not part of the repository - it is created once from
config.example.py and holds your OpenRouter key, which is why it is
listed in .gitignore. A fresh clone therefore does not have it, and
without this guard every entry point dies with

    ModuleNotFoundError: No module named 'config'

which tells a newcomer nothing at all about what to do. Every script you
can start calls require_config() before it imports config, and gets a
message naming the one command that fixes it.

This module deliberately imports nothing of its own - it has to work at
a point where config.py does not exist yet.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(BASE_DIR, "config.py")
VORLAGE = os.path.join(BASE_DIR, "config.example.py")

TEXT = """config.py is missing - this is the first step after cloning.

Copy the template once:

    Windows:   copy config.example.py config.py
    Linux/Mac: cp config.example.py config.py

Then open config.py and fill in two things:

    OPENROUTER_API_KEY   your key from https://openrouter.ai/keys
    USER_NAME            the name Alfred should call you

Everything else already has working defaults. config.py stays on your
machine - it is in .gitignore, so your key cannot end up in a commit."""


def require_config():
    """Stops with a readable message if config.py has not been created.

    Prints through Streamlit when a user interface is running, and to
    the console otherwise - detected from sys.modules, so that this
    module does not have to import Streamlit itself.
    """
    if os.path.exists(CONFIG):
        return

    fehlt_auch_die_vorlage = not os.path.exists(VORLAGE)
    text = TEXT
    if fehlt_auch_die_vorlage:
        text += (
            "\n\nconfig.example.py is missing too - you are probably not "
            "in the project directory. Change into the folder holding "
            "alfred.py and try again."
        )

    st = sys.modules.get("streamlit")
    if st is not None:
        st.error(text.replace("\n", "  \n"), icon="⚙️")
        st.stop()

    print("\n" + text + "\n", file=sys.stderr)
    raise SystemExit(1)


MEM0_FEHLT = """This tool works through mem0, which is an optional extra.

    pip install -r requirements-mem0.txt

mem0 is not needed for the chat (alfred.py) or for the viewer and the
consolidation - only for this tool, for alfred_ollama.py and for
db_cleanup.py."""

OLLAMA_FEHLT = """mem0 is installed, but it cannot reach Ollama.

mem0 computes a real embedding vector for every memory and uses a local
model for it, so Ollama has to be installed and RUNNING:

    https://ollama.com/download
    ollama pull nomic-embed-text

Check OLLAMA_HOST in config.py if Ollama listens somewhere other than
http://localhost:11434.

This is the price of the local route: nothing leaves your machine, but
the machine has to provide it. The chat itself (alfred.py) needs none
of this."""


def build_mem0_memory(mem0_config):
    """Builds mem0's Memory object, or stops with a readable message.

    Both failure modes look alarming raw: a missing mem0 produces an
    ImportError on a name nobody recognises, and a missing Ollama
    produces a connection traceback several frames deep inside mem0.
    Neither says what to do about it.
    """
    try:
        from mem0 import Memory
    except ImportError:
        print("\n" + MEM0_FEHLT + "\n", file=sys.stderr)
        raise SystemExit(1)

    try:
        return Memory.from_config(mem0_config)
    except Exception as e:
        # mem0 wraps the connection error in its own types, so the text
        # is the only reliable way to tell "Ollama is not there" from a
        # genuine configuration error - which is passed on unchanged.
        if "ollama" in str(e).lower() or "connect" in str(e).lower():
            print("\n" + OLLAMA_FEHLT + "\n", file=sys.stderr)
            raise SystemExit(1)
        raise
