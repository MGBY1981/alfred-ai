"""
alfred_sleep_ui.py

Alfred 2.0 - interface for memory consolidation

Start: streamlit run alfred_sleep_ui.py
(or start_alfred_sleep.bat)

This file contains NOTHING BUT presentation and operation. The complete
logic - database access, the three stages, applying a plan - lives in
alfred_sleep.py and is only called from here. The very same logic can
therefore be used unchanged from the command line, and the interface
cannot do anything the module cannot do as well.

The procedure is deliberately four-staged and cannot be short-cut:

    1. Inspect    - what is actually in the database?
    2. Plan       - read-only, costs a few cents, changes nothing
    3. Review     - every action individually selectable and editable
    4. Apply      - only here is anything written, with a backup beforehand

IMPORTANT: Close the Alfred chat before step 4. Two processes writing
into the same Chroma/SQLite file at the same time are an avoidable
source of errors.
"""

import os
from datetime import datetime

import streamlit as st

from config_guard import require_config
require_config()   # readable message if config.py is missing

import alfred_sleep as sleep
from config import USER_ID, USER_NAME
from memory_extractor_v5 import ALLOWED_TYPES

st.set_page_config(page_title="Alfred is sleeping", page_icon="🌙", layout="wide")


# ==========================================================
# Initialization
# ==========================================================

@st.cache_resource
def init_collection():
    return sleep.get_collection()


collection = init_collection()

for key, default in [
    ("plan", None),
    ("plan_pfad", None),
    ("ergebnis", None),
    ("bestaetigung", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ==========================================================
# Header and sidebar
# ==========================================================

st.title("🌙 Alfred is sleeping")
st.caption(
    "Consolidates the long-term memory: merges what belongs together, "
    "repairs placeholder subjects and lets expired things age. "
    "Nothing is written before you have seen and confirmed the plan."
)

with st.sidebar:
    st.subheader("Settings")

    user_id = st.text_input("Whose memory?", value=USER_ID).strip().lower()
    person = st.text_input(
        "Name for placeholder replacement",
        value=USER_NAME if user_id == USER_ID else user_id.capitalize(),
        help="Replaces phrasings like 'The user ...' in the memory text.",
    ).strip()

    st.divider()
    st.markdown("**Which stages should run?**")

    stufe1 = st.checkbox(
        "1 · Mechanical", value=True,
        help="Duplicates and placeholder subjects. Fixed rules, no model, "
             "cannot invent anything.",
    )
    stufe2 = st.checkbox(
        "2 · Merge", value=True,
        help="Find groups of facts that belong together and condense them "
             "into a single fact. Two model calls.",
    )
    stufe3 = st.checkbox(
        "3 · Age", value=True,
        help="Rewrite or remove time-bound facts that have expired.",
    )

    st.divider()
    st.markdown("**Models**")
    st.caption(
        "The chat is untouched by this – what follows applies to sleeping only. "
        "Even with the most expensive model a run costs no more than cents, "
        "because it runs rarely."
    )

    def modellwahl(label, optionen, default, hilfe):
        """Selection list with a free-text option.

        The lists live in alfred_sleep.py (SLEEP_MODELLE_*) - add new models
        there. For a one-off test, 'other …' is enough.
        """
        werte = list(optionen)
        if default not in werte:
            werte.insert(0, default)
        werte.append("other …")

        wahl = st.selectbox(
            label, options=werte,
            index=werte.index(default), help=hilfe,
        )
        if wahl == "other …":
            wahl = st.text_input(
                f"{label}: model ID", value=default,
                label_visibility="collapsed",
                placeholder="e.g. anthropic/claude-sonnet-5",
            ).strip() or default
        return wahl

    group_model = modellwahl(
        "Grouping", sleep.SLEEP_MODELLE_GRUPPIERUNG, sleep.SLEEP_GROUP_MODEL,
        "Finds memories that belong together. Pure recognition with numbers "
        "as output, but it carries the largest token load – an inexpensive "
        "model is the right choice here.",
    )
    merge_model = modellwahl(
        "Merging", sleep.SLEEP_MODELLE_URTEIL, sleep.SLEEP_MERGE_MODEL,
        "Phrases the consolidated fact out of a group. The hardest spot: "
        "judgement plus phrasing, and a detail that falls away leaves no "
        "trace.",
    )
    age_model = modellwahl(
        "Aging", sleep.SLEEP_MODELLE_URTEIL, sleep.SLEEP_AGE_MODEL,
        "Judges whether a time-bound fact has expired – sometimes with "
        "outright deletion as the consequence.",
    )

    if not (stufe2 or stufe3):
        st.caption("ℹ️ Stage 1 needs no model – the selection stays unused.")

    st.divider()
    vorhandene = []
    if os.path.isdir(sleep.PLAN_DIR):
        vorhandene = sorted(
            (f for f in os.listdir(sleep.PLAN_DIR) if f.endswith(".json")),
            reverse=True,
        )

    if vorhandene:
        st.markdown("**Load an earlier plan**")
        auswahl = st.selectbox("Plan file", options=["(none)"] + vorhandene,
                               label_visibility="collapsed")
        if auswahl != "(none)" and st.button("Load plan"):
            pfad = os.path.join(sleep.PLAN_DIR, auswahl)
            st.session_state.plan = sleep.lade_plan(pfad)
            st.session_state.plan_pfad = pfad
            st.session_state.ergebnis = None
            st.session_state.bestaetigung = False
            st.rerun()


stufen = tuple(n for n, an in [(1, stufe1), (2, stufe2), (3, stufe3)] if an)


# ==========================================================
# Step 1: Inspect the inventory
# ==========================================================

st.header("1 · Inventory")

try:
    memories = sleep.load_memories(collection, user_id)
except Exception as e:
    st.error(f"Database not readable: {e}")
    st.stop()

if not memories:
    st.warning(f"No memories found for '{user_id}'.")
    st.stop()

# A few key figures that show where something needs to be done.
platzhalter = sum(
    1 for m in memories
    if any(muster.search(m["text"]) for muster, _ in sleep._platzhalter_regeln(person or "X"))
)
zeitbezug = len(sleep.zeitbezogene(memories))
typ_other = sum(1 for m in memories if m["type"] == "other")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Memories", len(memories))
s2.metric("Placeholder subjects", platzhalter,
          help="They say 'The user ...' instead of the name. Stage 1 repairs that.")
s3.metric("With a time reference", zeitbezug,
          help="They contain a date, a month or a word like 'today'. "
               "Stage 3 checks whether they have expired.")
s4.metric("Type 'other'", f"{typ_other} / {len(memories)}",
          help="A high share means the typing says little.")

with st.expander("View all memories"):
    # ONE markdown block instead of one call per memory. With 676 entries
    # that used to be 676 separate Streamlit elements, which were rebuilt
    # on EVERY click anywhere on the page - even when the expander was
    # closed.
    st.markdown("\n".join(
        f"- **[{m['type']}]** {m['text']}  "
        f"<span style='color:gray;font-size:0.8em'>"
        f"({sleep.format_datum(m['created_at'])})</span>"
        for m in memories[::-1]
    ), unsafe_allow_html=True)


# ==========================================================
# Step 2: Generate the plan
# ==========================================================

st.header("2 · Put to sleep")

if not stufen:
    st.info("No stage selected – tick at least one on the left.")
else:
    spalte_a, spalte_b = st.columns([1, 3])

    with spalte_a:
        planen = st.button("🌙 Generate plan", type="primary", use_container_width=True)

    with spalte_b:
        st.caption(
            f"Runs with stage {', '.join(str(s) for s in stufen)}. "
            "A pure read operation – the database stays unchanged. "
            "Depending on the stage, the model calls cost a few cents."
        )
        if stufe2 or stufe3:
            st.caption(
                f"Merging/aging with `{merge_model}`"
                + (f" resp. `{age_model}`" if age_model != merge_model else "")
                + f", grouping with `{group_model}`."
            )

    if planen:
        anzeige = st.empty()
        try:
            plan = sleep.erzeuge_plan(
                collection, user_id, person=person, stufen=stufen,
                group_model=group_model,
                merge_model=merge_model,
                age_model=age_model,
                fortschritt=lambda t: anzeige.info(f"⏳ {t} …"),
            )
        except Exception as e:
            anzeige.empty()
            st.error(f"Planning failed: {e}")
            st.stop()

        anzeige.empty()
        st.session_state.plan = plan
        st.session_state.plan_pfad = sleep.speichere_plan(plan)
        st.session_state.ergebnis = None
        st.session_state.bestaetigung = False
        st.rerun()


plan = st.session_state.plan

if not plan:
    st.stop()


# ==========================================================
# Step 3: Review
# ==========================================================

st.header("3 · Review")

if plan.get("user_id") != user_id:
    st.warning(
        f"The loaded plan belongs to '{plan.get('user_id')}', "
        f"while '{user_id}' is selected. Please switch above or plan anew."
    )

kennzahlen = sleep.plan_kennzahlen(plan)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Actions", kennzahlen["aktionen_gesamt"])
k2.metric("Accepted", kennzahlen["aktionen_angenommen"])
k3.metric("To review", kennzahlen["zu_pruefen"])
k4.metric(
    "Afterwards", kennzahlen["erinnerungen_nachher"],
    delta=kennzahlen["erinnerungen_nachher"] - kennzahlen["erinnerungen_vorher"],
    help="Relative to the currently ticked actions.",
)

# Without this line you only read off the result of the PRESELECTION above
# and mistake it for the effect of the run. What the plan actually offers
# is stated here.
alle = kennzahlen["erinnerungen_nachher_alle"]
if alle != kennzahlen["erinnerungen_nachher"]:
    ersparnis = kennzahlen["erinnerungen_vorher"] - alle
    st.info(
        f"**If you accepted all {kennzahlen['aktionen_gesamt']} actions, "
        f"{alle} memories would be left** – {ersparnis} fewer than now "
        f"({ersparnis / max(kennzahlen['erinnerungen_vorher'], 1) * 100:.0f} %). "
        "The number above only shows what the current preselection achieves."
    )

plan_modelle = plan.get("modelle", {})
st.caption(
    f"Planning cost ${plan.get('kosten_planung', 0):.5f} "
    f"({plan.get('token_planung', 0)} tokens) · "
    f"merging `{plan_modelle.get('merging', '?')}`, "
    f"grouping `{plan_modelle.get('grouping', '?')}` · "
    f"saved under `{st.session_state.plan_pfad}`"
)

# --- Comparing two plans ----------------------------------------------
# The actual purpose of the model selection: have the same inventory
# planned twice and see where the models differ. A model that consistently
# finds less is not economical - it overlooks things.
if vorhandene:
    with st.expander("🔬 Compare this plan with another one"):
        andere = st.selectbox(
            "Comparison plan", options=["(none)"] + vorhandene, key="vergleich_wahl",
        )

        if andere != "(none)":
            plan_b = sleep.lade_plan(os.path.join(sleep.PLAN_DIR, andere))
            v = sleep.vergleiche_plaene(plan, plan_b)

            mb = plan_b.get("modelle", {})
            st.caption(
                f"**A** = current plan · merging "
                f"`{plan_modelle.get('merging', '?')}` · "
                f"{len(plan.get('aktionen', []))} actions ·  "
                f"**B** = `{andere}` · merging "
                f"`{mb.get('merging', '?')}` · "
                f"{len(plan_b.get('aktionen', []))} actions"
            )

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("In agreement", len(v["gleich"]),
                      help="Same sources, same result.")
            v2.metric("Different result", len(v["verschieden"]),
                      help="Both condense the same memories, "
                           "but phrase it differently.")
            v3.metric("Only A", len(v["nur_a"]))
            v4.metric("Only B", len(v["nur_b"]))

            if v["verschieden"]:
                st.markdown("**Same decision, different phrasing:**")
                for a_x, b_x in v["verschieden"]:
                    with st.container(border=True):
                        for q in a_x["quellen"]:
                            st.caption(f"· {q['text']}")
                        st.markdown(f"**A:** {(a_x.get('neu') or {}).get('text', '_(delete)_')}")
                        st.markdown(f"**B:** {(b_x.get('neu') or {}).get('text', '_(delete)_')}")

            for titel, eintraege in (("Found only by A", v["nur_a"]),
                                     ("Found only by B", v["nur_b"])):
                if not eintraege:
                    continue
                st.markdown(f"**{titel}:**")
                for x in eintraege:
                    ziel = (x.get("neu") or {}).get("text", "_(delete without replacement)_")
                    st.markdown(
                        f"- `{x['aktion']}` from {len(x['quellen'])} source(s): {ziel}"
                    )

if kennzahlen["zu_pruefen"]:
    st.warning(
        f"{kennzahlen['zu_pruefen']} action(s) are not pre-ticked – "
        "the model was unsure, the group contained several personal names, "
        "the result was shorter than the source, or something is to be "
        "deleted without replacement. You should look at those individually."
    )

# --- Filters and bulk actions ---
f1, f2, f3, f4 = st.columns([2, 1, 1, 1])

with f1:
    filter_stufe = st.multiselect(
        "Show stages",
        options=["mechanical", "merge", "age"],
        default=["mechanical", "merge", "age"],
        label_visibility="collapsed",
    )
with f2:
    nur_pruefen = st.checkbox("Only “to review”")
with f3:
    if st.button("Accept all visible", use_container_width=True):
        for a in plan["aktionen"]:
            if a["stufe"] in filter_stufe and (not nur_pruefen or a["pruefen"]):
                a["angenommen"] = True
        st.rerun()
with f4:
    if st.button("Reject all visible", use_container_width=True):
        for a in plan["aktionen"]:
            if a["stufe"] in filter_stufe and (not nur_pruefen or a["pruefen"]):
                a["angenommen"] = False
        st.rerun()

sichtbar = [
    a for a in plan["aktionen"]
    if a["stufe"] in filter_stufe and (not nur_pruefen or a["pruefen"])
]

# Pagination. Every action card costs about a dozen Streamlit elements
# (text field, type selector, several markdown lines). With close to 200
# actions plus the memory list, the page reached over 3000 elements, which
# were completely rebuilt on every single click - including the click on
# "Apply". That was exactly the supposed hang.
if "aktionsseite" not in st.session_state:
    st.session_state.aktionsseite = 1

if sichtbar:
    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        pro_seite = st.selectbox("per page", options=[25, 50, 100, "all"], index=1)
    if pro_seite == "all":
        seiten, aktionsseite = 1, 1
        seitenauswahl = sichtbar
    else:
        seiten = max(1, (len(sichtbar) + pro_seite - 1) // pro_seite)
        st.session_state.aktionsseite = min(max(1, st.session_state.aktionsseite), seiten)
        with b2:
            aktionsseite = st.number_input(
                "Page", min_value=1, max_value=seiten,
                value=st.session_state.aktionsseite, step=1, key="aktionsseite_eingabe",
            )
            st.session_state.aktionsseite = aktionsseite
        start = (aktionsseite - 1) * pro_seite
        seitenauswahl = sichtbar[start:start + pro_seite]
    with b3:
        st.caption(
            f"Page {aktionsseite} of {seiten} · "
            f"{len(sichtbar)} of {len(plan['aktionen'])} action(s) in the filter"
        )
else:
    seitenauswahl = []
    st.caption(f"Showing 0 of {len(plan['aktionen'])} action(s).")

STUFEN_FARBE = {"mechanical": "🔧", "merge": "🔗", "age": "⏳"}
typ_optionen = sorted(ALLOWED_TYPES)

for a in seitenauswahl:
    nr = a["nr"]

    with st.container(border=True):
        kopf = st.columns([0.05, 0.95])

        with kopf[0]:
            a["angenommen"] = st.checkbox(
                "accept", value=a["angenommen"],
                key=f"an_{nr}", label_visibility="collapsed",
            )

        with kopf[1]:
            marke = " · ⚠️ **review**" if a["pruefen"] else ""
            st.markdown(
                f"{STUFEN_FARBE.get(a['stufe'], '•')} **#{nr} · {a['aktion']}** "
                f"· {a['stufe']} · confidence {a.get('confidence', 1.0):.2f}{marke}"
            )

        st.markdown("**Before:**")
        for q in a["quellen"]:
            st.markdown(
                f"- `[{q['type']}]` {sleep.format_datum(q['created_at'])}: {q['text']}"
            )

        if a.get("neu"):
            st.markdown("**Afterwards:**")
            e1, e2 = st.columns([4, 1])
            with e1:
                a["neu"]["text"] = st.text_area(
                    "Text", value=a["neu"]["text"], key=f"txt_{nr}",
                    height=80, label_visibility="collapsed",
                )
            with e2:
                aktueller = a["neu"]["type"]
                a["neu"]["type"] = st.selectbox(
                    "Type", options=typ_optionen,
                    index=typ_optionen.index(aktueller) if aktueller in typ_optionen else 0,
                    key=f"typ_{nr}", label_visibility="collapsed",
                )
            st.caption(
                f"Date of the new entry: {sleep.format_datum(a.get('created_at'))} "
                f"(most recent date among the sources)"
            )
        else:
            st.markdown("**Afterwards:** _(will be deleted without replacement)_")

        if a.get("begruendung"):
            st.caption(f"Reason: {a['begruendung']}")


# ==========================================================
# Step 4: Apply
# ==========================================================

st.header("4 · Apply")

kennzahlen = sleep.plan_kennzahlen(plan)

sp1, sp2 = st.columns([1, 3])

with sp1:
    if st.button("💾 Save plan", use_container_width=True):
        st.session_state.plan_pfad = sleep.speichere_plan(plan)
        st.success(f"Saved: {st.session_state.plan_pfad}")

with sp2:
    st.caption(
        "Saves the current state of your selection as a new plan file – "
        "regardless of whether you apply it."
    )

mit_backup = st.checkbox(
    "Create a backup copy of the database before applying",
    value=True,
    key="mit_backup",
    help="Copies the complete chroma_db folder to sleep_backups/. "
         "That is the actual safety net – a JSON export would not be, "
         "because memory_import.py loses the dates when importing it "
         "back.",
)

if not kennzahlen["aktionen_angenommen"]:
    st.info("No action accepted – there is nothing to apply.")

elif not st.session_state.bestaetigung:
    st.markdown(
        f"**{kennzahlen['aktionen_angenommen']} action(s)** will be applied: "
        f"{kennzahlen['quellen_entfallen']} memory/memories fall away, "
        f"{kennzahlen['neue_eintraege']} new ones come into being. "
        f"**{kennzahlen['erinnerungen_vorher']}** become "
        f"**{kennzahlen['erinnerungen_nachher']}**."
    )
    st.caption("Please close the Alfred chat beforehand.")

    if st.button("Prepare applying", type="primary"):
        st.session_state.bestaetigung = True
        st.rerun()

else:
    st.warning(
        f"{kennzahlen['quellen_entfallen']} memory/memories will be deleted "
        f"from the database. That can only be undone via the backup copy."
    )
    b1, b2 = st.columns(2)

    with b1:
        if st.button("✅ Yes, apply now", type="primary", use_container_width=True):
            # The progress is reported block by block. What matters is not
            # just the display as such, but that you SEE that things are
            # moving on - an interface that stays mute for minutes looks
            # crashed, even when it is working cleanly.
            anzeige = st.empty()
            balken = st.progress(0.0, text="Applying is being prepared …")

            def zeige(text, anteil=None):
                anzeige.info(f"⏳ {text}")
                if anteil is not None:
                    balken.progress(min(max(anteil, 0.0), 1.0), text=text)

            try:
                ergebnis = sleep.apply_plan(
                    collection, plan, mit_backup=mit_backup, fortschritt=zeige,
                )
            except Exception as e:
                anzeige.empty()
                balken.empty()
                st.error(
                    f"**Applying aborted.** {type(e).__name__}: {e}\n\n"
                    "The database may be partially changed. If a backup copy "
                    "was created, it is located under "
                    "`sleep_backups/`."
                )
                st.exception(e)
                st.stop()

            anzeige.empty()
            balken.empty()
            st.session_state.ergebnis = ergebnis
            st.session_state.bestaetigung = False

            # Store the applied plan with a timestamp so that it stays
            # traceable what exactly was written.
            plan["angewendet_am"] = datetime.now().isoformat()
            st.session_state.plan_pfad = sleep.speichere_plan(plan)
            st.rerun()

    with b2:
        if st.button("✖ Cancel", use_container_width=True):
            st.session_state.bestaetigung = False
            st.rerun()


ergebnis = st.session_state.ergebnis

if ergebnis:
    dauer = ergebnis.get("dauer") or 0
    if ergebnis.get("fehler"):
        st.warning(
            f"Finished with errors after {dauer:.1f}s: "
            f"{ergebnis['angelegt']} created, {ergebnis['geloescht']} deleted, "
            f"**{len(ergebnis['fehler'])} block(s) failed**."
        )
    else:
        st.success(
            f"Done in {dauer:.1f}s: {ergebnis['angelegt']} entry/entries "
            f"created, {ergebnis['geloescht']} deleted."
        )

    if ergebnis.get("backup"):
        st.caption(f"Backup copy: `{ergebnis['backup']}`")
    else:
        st.caption("⚠️ Applied without a backup copy – no way back.")

    for f in ergebnis.get("fehler", []):
        st.error(f)
    if ergebnis.get("fehler"):
        st.caption("All runs are additionally recorded in `sleep_backups/apply.log`.")
    st.caption(
        "The numbers above under “Inventory” will update the next time the "
        "page is reloaded."
    )
