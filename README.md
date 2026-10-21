<img src="emojis/robot_coffee.png" width="120" align="left" alt="Alfred, a small grey robot butler holding a cup of coffee">

# Alfred

**A personal assistant that remembers you — and lets you manage it's memories

Runs on your machine, stores everything locally, and takes about five
minutes to set up.

<br clear="left">

---

Most assistants with a memory only ever add to it. That works
beautifully for a fortnight. Then you have six entries saying you like
pizza, a dozen fragments that just read "LEGO", and a note about a
dentist appointment from last March.

Alfred is my attempt at the other half of the problem: not just
remembering, but keeping the memory **clean, readable and yours**. He
has a consolidation pass that merges duplicates, drops what has expired
and rewrites fragments into real sentences — and it never changes
anything behind your back. It writes a plan, shows you every single
action, and waits.

It is a personal project, built over a few months and used every day. It
is not a framework and does not want to be: it is a working assistant
you can clone, read end to end, and bend to your taste.

## What makes it different

**It thinks about what is worth remembering.** Not everything you type
becomes a memory. Small talk, weather, passing moods and questions are
rejected on purpose. Facts get split into atomic sentences, each with a
subject, so that "I like LEGO, Batman and Star Wars" does not end up as
three orphaned words. And you stay in charge of the rule: say *"you
don't need to remember that"* and nothing from that exchange is stored —
say *"please remember this"* and it is kept even if it would normally be
filtered out.

**It only sends the memories that matter.** Once there are more
memories than fit comfortably in a prompt, a cheap model preselects. The
trick is what it is allowed to answer: it sees the memories *numbered*
and may reply with **nothing but ID numbers**. It can therefore never
reword a memory, lose half a sentence or invent one — and it needs no
translation, because it never touches text at all.

**Several people, one Alfred.** Memories are partitioned per user, and
every tool honours it — the chat asks who you are at startup, the viewer
and the consolidation let you pick whose memory you are working on. Your
partner can use the same installation without ever seeing your facts,
and without you seeing theirs.

**Nothing changes without your say-so.** The consolidation is split in
two: producing a plan is strictly read-only, applying it is a separate
step that backs the database up first and aborts if the backup fails.
Anything the model was unsure about is flagged and *not* preselected.

**It is honest about cost.** Under every answer you can see how long
each stage took and what the message really cost — all three calls
itemised, not just the answer, straight from OpenRouter's own figures. A
week of daily use is a few cents, and you can watch where they go.

**Setup is three commands and an API key.** No local model, no GPU, no
Docker, no account anywhere except OpenRouter.

## Quickstart

You need **Python 3.10 or newer** and an **OpenRouter API key** (free to
create, pay-as-you-go, costs cents: <https://openrouter.ai/keys>).

**1. Get the code**

```bash
git clone https://github.com/MGBY1981/alfred-ai.git
cd alfred-ai
```

**2. Install the three dependencies**

```bash
pip install -r requirements.txt
```

That is streamlit, chromadb and requests. Nothing else — everything that
needs an AI model goes to OpenRouter over HTTPS.

**3. Create your configuration**

```bash
copy config.example.py config.py     # Windows
cp config.example.py config.py       # Linux / macOS
```

**4. Give Alfred your API key**

Two ways. Pick one — the first needs no editing at all.

*A — as an environment variable (recommended)*

```bat
setx OPENROUTER_API_KEY "sk-or-v1-your-key-here"
```

```bash
echo 'export OPENROUTER_API_KEY="sk-or-v1-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

The first is Windows, the second Linux/macOS (use `~/.zshrc` on a
current Mac). Set once, works for good, and the key lives in no file in
this project. **On Windows you then need a new terminal window** —
`setx` writes the value for *future* processes, so anything already
running will not see it. Note that this is `setx`, not `set`: `set` only
lasts for the one window it was typed in.

Then start Alfred. Your `config.py` can stay exactly as it was copied —
it reads the variable by itself, and the chat asks for your name on the
first screen anyway.

*B — write it into config.py*

If you would rather not touch environment variables, open `config.py`
and put the key in directly:

```python
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-your-key-here")
USER_NAME = "Anna"      # optional, Alfred asks if you leave it empty
```

`config.py` is listed in `.gitignore`, so your key cannot end up in a
commit by accident. Do **not** put the key into one of the `.bat` files
instead — those *are* tracked by git, and it would be committed.

The environment variable wins if both are set.

**5. Start talking**

```bash
streamlit run alfred.py
```

or double-click `start_alfred.bat` on Windows. Your browser opens at
<http://localhost:8505> and Alfred asks for your name.

That is it. Tell him something about yourself, reload the page, and ask
him what he knows about you.

**The other three tools**, once you have a few memories:

```bash
streamlit run memory_viewer.py      # browse and delete      :8506
streamlit run alfred_sleep_ui.py    # the consolidation      :8507
streamlit run alfred_dialog_ui.py   # two AIs talking        :8508
```

Each has a `start_*.bat` next to it.

**Trying the local route without committing to it.** `start_alfred_ollama.bat`
sets two environment variables before it starts Alfred:

```bat
set ALFRED_EXTRACTOR_BACKEND=ollama
set ALFRED_FILTER_MODE=ollama
```

They apply to that one window and nothing else — `config.py` is not
touched, `start_alfred.bat` is unaffected, and closing the window undoes
it. Drop one of the two lines to go half local, say a cloud filter with
local extraction. The sidebar shows where each stage ended up, so you
can see what you got.

The same variables work for the measurement scripts, which is the
cheapest way to compare the two:

```bash
python memory_extractor_regression_test_v5.py                 # cloud model
ALFRED_EXTRACTOR_BACKEND=ollama python memory_extractor_regression_test_v5.py
```

The second run scores your local model on the same 43 cases, and each
run prints the model it used next to the result.

**Reaching Alfred from your phone.** By default everything listens on
`localhost` only — nothing on your network can connect. If you want to
open the chat from another device at home, start it like this instead:

```bash
streamlit run alfred.py --server.port=8505 --server.address=0.0.0.0
```

Be aware of what that means: **there are no passwords.** Anyone who can
reach that address can type any name and read that person's memories.
Fine on a home network you trust, a bad idea on café or hotel wifi.
Note that leaving the address out entirely is *not* the safe option —
Streamlit then binds to all interfaces as well, so `localhost` has to be
stated explicitly, which is what the `.bat` files do.

## Running the memory work on your own machine

Three of the four stages can run locally through
[Ollama](https://ollama.com/download). The answer cannot — that one is
always a cloud model.

```bat
ollama pull llama3.1:8b        for the extraction
ollama pull nomic-embed-text   for the embeddings
ollama pull llama3.2           for the filter
```

Then `start_alfred_ollama.bat`, or the two environment variables from
the quickstart. Both models have to be pulled *before* the first run;
Ollama does not fetch them on demand here.

**What you get.** Two things, and it is worth being clear which matters
to you. The facts about you are distilled and selected at home, so no
provider ever sees the raw material of your memory — only the
conversation itself goes out. And the running cost drops to just the
answer model: the filter and the extractor stop billing entirely.

**What it costs you.** Wall-clock time, and roughly an order of
magnitude on the two stages that moved. One measurement on a reasonably
current laptop, same question both ways:

| | cloud | local |
|---|---|---|
| Memory filter | 0.69 s | 5.97 s |
| Memory extraction | 0.69 s | 9.16 s |
| Answer model | cloud either way | cloud either way |

Only one of those two is a wait, though. Extraction runs *after* the
answer is on screen (`MEMORY_EXTRACTION_DEFERRED`), so its nine seconds
cost you nothing you can feel. The filter runs before the answer, so
that one you do sit through. In practice the message felt about twice as
slow end to end, not ten times.

**And quality may drop.** An 8B model is not a frontier model. The
extraction prompt is demanding — atomic facts, explicit subjects, no
inference, honour an explicit "don't remember that" — and a smaller
model holds that line less reliably. Do not take that on trust either
way:

```bash
python memory_extractor_regression_test_v5.py                            # cloud
ALFRED_EXTRACTOR_BACKEND=ollama python memory_extractor_regression_test_v5.py
```

Same 43 cases, two models, and each run prints which model produced the
score.

**About the model choice.** `llama3.1:8b` for the extraction is a
measured decision: the regression test showed the smaller `llama3.2`
hallucinating on small talk and occasionally returning the prompt's own
example as its answer. The 8B model is there because the 3B one failed.

`llama3.2` on the filter, by contrast, is simply what was configured
back when the filter ran locally, and has never been measured against
anything. There is an argument for a small model in that spot — the
filter only ever answers with ID numbers, which is a mechanical job, and
it is the call that sees the most tokens, so speed counts for most
there. But an argument is not a measurement, and `filter_compare.py`
only compares cloud models today. If you try a different local model on
the filter, I would be glad to hear what you find.

## The consolidation, close up

This is real output from `python alfred_sleep.py --report plan.json`:

```
Memories before : 800
Actions         : 3 (1 accepted, 2 to review)
Memories after  : 798 (-3 / +1)

------------------------------------------------------------------------
[x] #12  MERGE  (stage: merge, confidence: 0.9)
      - [interest] 14.03.2026: Italy
      - [interest] 14.03.2026: Croatia
      - [interest] 14.03.2026: Greece
    = [interest] 14.03.2026: Anna likes travelling to the Mediterranean,
                             among other places to Italy and Croatia.
      Reason: Six fragments of an enumeration combined into one sentence.
------------------------------------------------------------------------
[ ] #13  MERGE  (stage: merge, confidence: 0.6)  <-- REVIEW
      - [interest] 14.03.2026: Anna is interested in Python.
      - [interest] 02.04.2026: Anna's husband uses Moodle.
    = [interest] 02.04.2026: Anna's husband is interested in Python.
      Reason: The source texts name different subjects.
------------------------------------------------------------------------
[ ] #14  DELETE  (stage: age, confidence: 0.95)  <-- REVIEW
      - [interest] 14.03.2026: Anna has an appointment at the doctor
                               on 14 March.
    = (will be deleted without replacement)
      Reason: The date is in the past; the appointment has happened.
------------------------------------------------------------------------
```

`[x]` is preselected, `[ ]` is not. Action 13 is the one worth catching:
two memories about two different people, which a careless merge would
have fused into a single wrong fact. Nothing in that list happens until
you press Apply — and the same list is clickable in the browser.

It is called "sleep" after what it is modelled on. One reason humans
sleep is to consolidate memories: deciding what belongs together, and
what can go.

## Measure it yourself

Most of what you read about assistants like this one is assertion. Here
are the tools to check instead, pointed at your own data and your own
models.

**43 test cases for the fact extraction.**

```bash
python memory_extractor_regression_test_v5.py
```

43 German conversations, each with the memories it should and should not
produce, scored PASS / WARNING / TYPE FAIL / FAIL with per-category
statistics and an overall percentage. It runs against whichever
extractor backend you configured, and prints the model it used next to
the result.

You will not find that percentage quoted here, on purpose. It depends on
the model, the prompt and the language, so a number in a README is stale
the moment any of the three moves — and a stale number is worse than
none. Run it and you have today's figure for your setup. It is also how
I learned that a 3B model hallucinates on small talk and occasionally
returns the prompt's own example as its answer.

Cases where the backend could not be reached are reported separately and
left out of the score, so a flaky connection cannot quietly look like a
quality problem.

**Which filter model is worth it.**

```bash
python filter_compare.py
```

Runs the same question through several models against *your* memories
and prints what each selected, how long it took and what it cost, down
to fractions of a cent. The table in `config.example.py` is one such run
and is why the default is Flash rather than its cheaper Lite sibling,
which waved career facts through as hobbies.

**Does extraction work in your language?**

```bash
python extraction_check.py
```

Eight matched statements in English and German — four that should be
stored, four that should not — for a fraction of a cent.

One number I will quote, because it is a property of the code rather
than of any model: applying a consolidation plan of 150 actions over 800
memories takes **0.58 s instead of 6.75 s** since the writes are
batched.

## What is in here

| File | What it does | Port |
|---|---|---|
| `alfred.py` | **The chat. Start here.** Needs only an OpenRouter key. | 8505 |
| `alfred_ollama.py` | Same chat, writing through mem0; needs a local Ollama. | 8505 |
| `memory_viewer.py` | Browse, search, filter and delete memories. | 8506 |
| `alfred_sleep_ui.py` | The consolidation, with the reviewable plan. | 8507 |
| `alfred_dialog_ui.py` | Two assistants talking to each other. | 8508 |
| `alfred_sleep.py` | The consolidation logic, also usable from the shell. | — |
| `memory_export.py` / `memory_import.py` | JSON export and import, hand-editable in between. | — |
| `filter_compare.py` | Measures filter models against your own memories. | — |
| `extraction_check.py` | Probes whether extraction works in your language. | — |
| `language_packs.py` | The language-dependent word lists (German, English). | — |
| `memory_extractor_regression_test_v5.py` | The 43 test cases. | — |

## Language

```python
ASSISTANT_LANGUAGE      = "en"
ASSISTANT_LANGUAGE_NAME = "English"
```

Any language works for the chat and the memory, because the name goes
into the prompt verbatim. Four pieces of logic additionally work on the
text locally without asking a model — subject detection, fragment
detection, the time-reference prefilter and rewriting "the user" into
your name — and those need word lists. **German and English ship
complete.** Choose a language without a pack and the chat and the memory
work normally while those four switch themselves off and say so, rather
than searching Italian text for German words. Adding a language means
translating four lists in `language_packs.py`.

German is the language the extraction quality was measured in. The
heuristics were written against it specifically: German capitalises
every noun, which breaks the naive "a capitalised word must be a name"
approach that would otherwise flag nearly every group as suspicious.

If you are unsure whether extraction holds up in your language, there is
a quick probe:

```bash
python extraction_check.py
```

It runs the same eight statements in English and German — four that
should be stored, four that should not — and prints what came back for
each, so an asymmetry is visible in one screen for a fraction of a cent.
It also flags any memory written under a name other than the speaker's.

## Two AIs talking

`alfred_dialog_ui.py` lets Alfred talk to a second assistant for a fixed
number of turns, each side using its own memory, with a hard turn limit
and a cost cap. It is the least useful and most interesting thing here.

By default Alfred stores **nothing** from such a conversation — the
class representing him in a dialogue has no write path at all. Switch it
on and you get a warning, and everything he files is marked twice: the
text says who said it ("According to Alfred Remote, …") and the metadata carry
`source = "dialog:<id>"`, so you can find it all again and remove it in
one go. Be clear on what that is and is not: once saved, hearsay
memories are used like any other. The protection is the off-by-default
switch and the marking, not a quarantine.

**The counterpart is not in this repository.** Nothing in the code cares
what is on the other end — it speaks plain HTTP, so a second computer, a
container, or this same machine on another port all work equally well.
Mine happens to be a small Java service on a Raspberry Pi, which is
where the idea of an always-on second assistant came from, but that is a
detail of my setup and not a requirement. Any server answering these two
endpoints will do:

- `GET /health` — is anyone home
- `POST /chat` — receives `conversation_id`, `partner_name`, `role`,
  `topic`, `use_memory`, `save_memory`, `web_search` and a `history` of
  `{role: "self"|"partner", content}`, and answers with `text`, `model`,
  `cost` and optionally counts of the memories it used and saved

Both use `Authorization: Bearer <token>`. `python alfred_dialog.py
--health` tells you whether the other end is reachable.

## Honest limitations

- **No passwords.** Separate memories per user is a partition, not a
  security boundary — anyone who can open the page can type any name.
  Fine for a household, not for strangers.
- **There is no fully offline mode.** Extraction, filtering and the
  embeddings can all be moved onto your own machine with Ollama, so the
  decisions about your memory happen at home — but the answer itself is
  always written by a cloud model. The sidebar of the running chat shows
  which stage goes where, with a 🏠 or a ☁ in front of each.
- **`alfred_ollama.py` needs a running Ollama** and says so in the page
  if it cannot reach one. That is the point of that variant;
  `alfred.py` is the one that works without it.
- **The vectors in the database are meaningless.** `alfred.py` writes
  zero vectors, because nothing here searches by similarity — grouping
  for the consolidation is done by model judgement instead. Point
  something vector-based at this database and it will find little.
  `memory_import.py` can recompute real embeddings.
- **The consolidation costs real money** with the recommended models,
  because judging what may be merged is the one place a weak model does
  damage. A plan over a few hundred memories is cents, not euros, and
  the models are configurable.
- **Extraction quality outside German is unmeasured.** The prompt is
  language-neutral and the English pack is complete, but I have not run
  43 test cases against it.

## Licence

Code: MIT, see [LICENSE](LICENSE).

The 18 robot emoji in `emojis/` are AI-generated and free to use without
restriction or attribution — see [emojis/CREDITS.md](emojis/CREDITS.md).

Dependencies: streamlit, chromadb and requests, all Apache-2.0. mem0 is
optional and needed by three files only.
