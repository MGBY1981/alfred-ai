"""
config.example.py

Central configuration for Alfred 2.0.

GETTING STARTED
---------------
This file is a TEMPLATE. Copy it once:

    Windows:  copy config.example.py config.py
    Linux:    cp config.example.py config.py

and then fill in your OpenRouter key and your name below.
Every component (web chat, memory tools, tests) reads its settings
exclusively from config.py.

Your real config.py is listed in .gitignore and is never committed -
that way your key cannot end up in a commit by accident.

THE BARE MINIMUM YOU HAVE TO FILL IN
------------------------------------
    OPENROUTER_API_KEY   without it, Alfred cannot answer at all
    USER_NAME            how Alfred should address you

Everything else comes with usable defaults.
"""

import os

# --------------------------------------------------------
# Connection to the second assistant (optional)
# --------------------------------------------------------
# Only needed if you want to use alfred_dialog_ui.py - the feature that
# lets two AI assistants talk to each other. It requires a dialog server
# running on the other end - a second computer, a container, or this
# same machine on another port. A Raspberry Pi works nicely, but
# nothing in the code cares what it is.
#
# The address points to a machine on YOUR network, and the token has to be
# identical on both sides. Leave these empty if you don't use the feature.

DIALOG_PARTNER_URL = "http://your-second-machine:8765"
DIALOG_PARTNER_TOKEN = ""

# --------------------------------------------------------
# Base directories
# --------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOJI_DIR = os.path.join(BASE_DIR, "emojis")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# The name of the collection inside the Chroma database. Everything -
# chat, viewer, consolidation, export - works on this one collection, so
# changing it gives you a second, separate memory to play with while the
# real one stays untouched. The mem0 section at the very bottom picks
# this value up, so the two can never drift apart.
CHROMA_COLLECTION = "alfred_memory"

# --------------------------------------------------------
# Language
# --------------------------------------------------------
# Which language Alfred answers in and stores memories in. Set BOTH
# values together.
#
#     ASSISTANT_LANGUAGE       picks the language pack (see below)
#     ASSISTANT_LANGUAGE_NAME  goes into the prompts, verbatim
#
# ANY language works for the chat and for storing memories: the name is
# simply put into the prompt, so "Spanish" or "Polish" need nothing else.
# The memory filter is unaffected either way - it only ever returns ID
# numbers, never text, so it is language-independent by design.
#
# FULL support, however, means a bit more. Four pieces of logic work on
# the text locally, without asking a model, and they need to know the
# language: subject detection, fragment detection, the time-reference
# pre-filter of alfred_sleep stage 3, and the "the user" -> name
# rewriting. Those live in language_packs.py, and packs ship for:
#
#     "de" / "German"      complete, and the language the extraction
#                          quality was measured in (see below)
#     "en" / "English"     complete
#
# Pick a language with no pack (say "it" / "Italian") and the chat and
# the memory work normally - the four local heuristics switch themselves
# OFF rather than searching Italian text for German words. alfred_sleep
# then says so in its progress line and stage 3 finds nothing. Writing a
# pack means translating four word lists; language_packs.py explains how.
#
# Note on the regression test: memory_extractor_regression_test_v5.py
# ships with 43 GERMAN test cases. It overrides the language to German on
# its own, whatever is set here, so it keeps working unchanged - and it
# measures whichever extractor backend you configured below.

ASSISTANT_LANGUAGE = "en"
ASSISTANT_LANGUAGE_NAME = "English"

# --------------------------------------------------------
# User
# --------------------------------------------------------

# USER_NAME is the name Alfred addresses you by, and the name he files
# facts about you under ("Anna likes pizza."). USER_ID is that same name
# in lower case and serves as the key in the database.
#
# The web chat asks for your name on startup anyway - these values are
# only the default there, plus the fallback for the command-line tools
# (memory_export.py, alfred_sleep.py, filter_compare.py).

USER_NAME = ""
USER_ID = USER_NAME.lower()

# Where you live. Alfred mentions this in the system prompt so that he knows
# which place you mean when you ask about weather or the news. Leave it empty
# if you'd rather not share it - he then simply drops the location reference.

USER_LOCATION = ""

# --------------------------------------------------------
# OpenRouter - access
# --------------------------------------------------------

# Create a key at https://openrouter.ai/keys
#
# Two ways to supply it; the first one is the safer choice:
#
#   1. Set an environment variable - the key then stays out of this
#      file entirely:
#         Windows:  setx OPENROUTER_API_KEY "sk-or-v1-..."
#         Linux:    export OPENROUTER_API_KEY="sk-or-v1-..."
#
#      On Windows this is setx, NOT set: setx stores the value
#      permanently, but only for processes started afterwards - so
#      open a new terminal before you start Alfred. `set` by contrast
#      lasts only for the one window it was typed in.
#
#   2. Enter it directly below. That is more convenient, but it leaves a
#      valid key in plain text on your disk - if you go this route, don't
#      share this file with anyone and don't upload it anywhere.
#
# The environment variable takes precedence if both are set.

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = "http://localhost:8505"
OPENROUTER_TITLE = "Alfred Assistant"

# --------------------------------------------------------
# OpenRouter - model choice
# --------------------------------------------------------

# The premium model that writes the actual answer. This is the one place
# where spending a bit is worth it - it is the voice the user hears.
#
# The tilde in "~anthropic/..." is NOT a typo: OpenRouter reads it as
# "always the latest version of this model family".
#
# Alternatives:
#   "~anthropic/claude-haiku-latest"   currently Haiku 4.5, approx. $1 / $5 per 1M tokens
#   "anthropic/claude-haiku-4.5"       pin to one fixed version
#   "google/gemini-2.5-flash"          cheaper, but deprecated on 16.10.2026
#   "~google/gemini-flash-latest"      current Flash generation, approx. $1.50 / $7.50

ANSWER_MODEL = "~anthropic/claude-haiku-latest"

# How many web search results the answer model gets to work with.
ANSWER_WEB_SEARCH_MAX_RESULTS = 3
ANSWER_WEB_SEARCH_CONTEXT_SIZE = "low"

ANSWER_TIMEOUT = 60

# --------------------------------------------------------
# Memory filtering (which memories are relevant to the question?)
# --------------------------------------------------------

# Modes:
#   "off"        - NO filtering at all. Every memory goes straight into the
#                  answer model's system prompt. The fastest option, and the
#                  best one for quality, as long as the database stays small.
#                  THE RIGHT CHOICE WHILE YOU ARE STILL BELOW ROUGHLY A
#                  HUNDRED FACTS - which is where a fresh clone starts.
#   "openrouter" - A cheap cloud model preselects the relevant memories.
#                  Worth it once you reach several hundred facts.
#   "ollama"     - The old local route via memory_manager_v2_1.py.
#                  It works, but it is the bottleneck (20-30 s).

# The environment variable wins if it is set, so one run can take the
# local route without this file being touched - that is what
# start_alfred_ollama.bat does. Change the value here to make it stick.
MEMORY_FILTER_MODE = os.getenv("ALFRED_FILTER_MODE", "openrouter")

# Safety net: below this many memories, nothing is filtered, no matter what
# the setting above says. With only a few facts, filtering is pure waste -
# it costs time and money and risks losing something important. Raise it if
# your database grows and you still want to send everything along; lower it
# to let the filter kick in sooner.

MEMORY_FILTER_THRESHOLD = 1

# Model for MEMORY_FILTER_MODE = "openrouter".
#
# Real-world comparison (30.07.2026, filter_compare.py, 82 genuine memories,
# question "What hobbies does our family have?"):
#
#   Model                             Selected  Time    Cost
#   google/gemini-2.5-flash-lite      35 / 82   1.54s   $0.00026
#   google/gemini-2.5-flash           35 / 82   1.63s   $0.00101
#   ~anthropic/claude-haiku-latest    19 / 82   2.18s   $0.00229
#
# Flash-Lite IS OUT: on this question it wrongly waved through career and
# project facts ("<user> is a software developer.", "Claude AI", "AI butler")
# as hobbies - Flash (the full model) and Haiku both did not.
#
# Haiku's 19 hits were the leanest selection, but that is NO quality
# advantage: in the process it missed a genuine fact (an interest of a
# family member) that both Flash models picked up correctly - probably
# because the stored text was awkwardly worded (see the TODO in
# memory_extractor_v5.py about the fragment/subject check).
#
# Verdict: Flash (NOT Lite) hits the sweet spot for this job - as accurate
# as Haiku, without its misclassification, for about a third of the cost.
#
#   "google/gemini-2.5-flash"        approx. $0.30 / $2.50 - CURRENT CHOICE
#   "google/gemini-2.5-flash-lite"   approx. $0.10 / $0.40 - noticeably too imprecise
#   "~anthropic/claude-haiku-latest" the eventual "upgrade", in case Flash
#                                    does start to struggle as the database grows
#
# WATCH OUT when picking a model: the CURRENT Gemini Flash generation (3.5/3.6)
# costs around $1.50 per 1M input tokens, which is MORE than Haiku 4.5 ($1).
# The 2.5 generation (used here) is not affected by that, but it is being
# deprecated on 16.10.2026 - migrate to a successor before then.

MEMORY_FILTER_MODEL = "google/gemini-2.5-flash"
MEMORY_FILTER_TIMEOUT = 30

# --------------------------------------------------------
# Memory extraction (which new facts does the message contain?)
# --------------------------------------------------------

# Backends:
#   "ollama"     - llama3.1:8b, running locally. Free, but slow (this is the
#                  real time sink, not the filtering).
#   "openrouter" - A cloud model. Faster and, as a rule, more accurate than
#                  an 8B model. Costs a fraction of a cent per message.
#   "off"        - Store no new memories at all (useful for testing).
#
# The prompt from memory_extractor_v5.py stays IDENTICAL in every case -
# all that changes is who answers it. Your regression test therefore stays
# valid and can measure the two backends against each other.

# Same as MEMORY_FILTER_MODE above: ALFRED_EXTRACTOR_BACKEND overrides
# this for a single run without editing anything.
MEMORY_EXTRACTOR_BACKEND = os.getenv("ALFRED_EXTRACTOR_BACKEND", "openrouter")

MEMORY_EXTRACTOR_CLOUD_MODEL = "google/gemini-2.5-flash-lite"
MEMORY_EXTRACTOR_TIMEOUT = 60

# Run the extraction only AFTER the answer has been shown. That way the user
# never waits on the storage logic. Set this to False if you want the log to
# show the steps in strict order while debugging.
MEMORY_EXTRACTION_DEFERRED = True

# ========================================================================
# OPTIONAL: running locally with Ollama
# ========================================================================
# EVERYTHING BELOW CAN BE IGNORED unless you want the local route. Out of
# the box nothing here is used: alfred.py talks to OpenRouter and writes
# to Chroma directly, with no local model anywhere.
#
# What the local route gets you: the decisions ABOUT YOUR MEMORY happen
# on your own machine. Which facts get extracted from a message, which
# memories are selected for a question, and the embedding vectors - all
# of that stays at home, costs nothing and is not seen by anyone.
#
# WHAT IT DOES NOT GET YOU: the conversation itself. The answer is
# always written by ANSWER_MODEL through OpenRouter - there is no local
# option for it, because that is the one place where model quality is
# immediately audible. So "fully offline" is not something this project
# offers, and you should not read it as such.
#
# Three switches, independent of each other:
#
#   MEMORY_EXTRACTOR_BACKEND = "ollama"   extraction runs locally
#   MEMORY_FILTER_MODE       = "ollama"   filtering runs locally
#   start alfred_ollama.py instead of alfred.py
#                                         storage runs through mem0
#
# The models you need for them:
#
#   ollama pull llama3.1:8b        for the extraction
#   ollama pull nomic-embed-text   for the embeddings (alfred_ollama.py)
#
# Mixing is the interesting case: local extraction with a cloud answer
# model means the facts about you are only ever distilled at home, while
# the conversation still gets a good model. The sidebar of the running
# chat shows which stage goes where, so you can see what you picked.

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
MEMORY_MANAGER_MODEL = "llama3.2"

# Number of CPU threads for Ollama inference. Set this to your count of
# physical cores (not logical cores / hyperthreads) - under a pure matrix
# workload, the extra hyperthreads tend to hold things back rather than
# help. On a CPU with P- and E-cores (an Intel Core Ultra, say), it can be
# worth trying even lower values (P-cores only) to see whether that beats
# using every physical core.
OLLAMA_NUM_THREAD = 10

# For fact extraction (memory_extractor_v5.py), the regression test shows
# llama3.2 (3B) is not reliably good enough (hallucinations on small talk,
# and it sometimes returns the example from the prompt as its answer).
# Install the bigger model before use with: `ollama pull llama3.1:8b`.
# Then run the regression test again and compare the score to llama3.2.
MEMORY_EXTRACTOR_MODEL = "llama3.1:8b"

# How many of the most recent chat messages (the short-term memory of the
# current session) are sent along to the answer model with each request.
MAX_CHAT_HISTORY_MESSAGES = 50

# --------------------------------------------------------
# Diagnostics
# --------------------------------------------------------

# Shows below every answer how long each stage took and what the call cost
# according to OpenRouter. Very useful while tuning; for everyday use, feel
# free to set this to False.
SHOW_TIMINGS = True

# --------------------------------------------------------
# mem0 (part of the optional local route above)
# --------------------------------------------------------
# Read by exactly three files - alfred_ollama.py, memory_import.py and
# db_cleanup.py. Everything else gets the collection name from
# CHROMA_COLLECTION at the top and never looks in here.
#
# Of the two models below, only the EMBEDDER is ever used. Memories are
# written with infer=False, so mem0 stores the text as given instead of
# asking a model what to make of it - deciding what is worth keeping is
# the extractor's job, and reading goes straight through Chroma. The llm
# entry exists because mem0's configuration wants one; no code path in
# this project calls it, so there is no need to pull llama3.2 for it.
#
# That also explains why the local route feels fast: nomic-embed-text
# turns one short sentence into a vector in milliseconds. The slowness
# people associate with Ollama comes from generating text with a chat
# model, and that never happens here.

MEM0_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": CHROMA_COLLECTION,
            "path": CHROMA_DB_DIR,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": OLLAMA_CHAT_MODEL,
            "temperature": 0,
            "max_tokens": 1000,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": OLLAMA_EMBED_MODEL,
        },
    },
    "version": "v1.1",
}
