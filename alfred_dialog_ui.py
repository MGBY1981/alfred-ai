"""
alfred_dialog_ui.py

Alfred 2.0 - user interface for the conversation between two AI systems

Start: streamlit run alfred_dialog_ui.py
(or start_alfred_dialog.bat, port 8508)

As with alfred_sleep_ui.py, this file contains nothing but presentation and
controls; the logic lives entirely in alfred_dialog.py.

WHY ONE TURN PER RUN
--------------------
The conversation loop does NOT run through in one go. Instead exactly one
turn is executed per Streamlit run and a rerun is triggered afterwards. That
costs a little elegance in the code, but it is the only way in which stopping
and interjecting in the middle of the conversation actually work - and that is
what this is about: you are supposed to be able to watch and intervene, not
to read a finished transcript afterwards.
"""

import os
from datetime import datetime

import streamlit as st

from config_guard import require_config
require_config()   # readable message if config.py is missing

import alfred_dialog as dlg
from config import USER_NAME

st.set_page_config(page_title=f"Alfred talks to {dlg.DIALOG_NAME_LITE}", page_icon="🗣️", layout="wide")


for schluessel, standard in [
    ("dialog", None),
    ("laeuft", False),
    ("pfad", None),
    ("einwurf_offen", False),
]:
    if schluessel not in st.session_state:
        st.session_state[schluessel] = standard


st.title(f"🗣️ {dlg.DIALOG_NAME_ALFRED2} talks to {dlg.DIALOG_NAME_LITE}")
st.caption(
    "Two AI systems talk to each other, each with its own memory. "
    f"{dlg.DIALOG_NAME_LITE} answers from its own machine and is allowed to learn as it goes – "
    f"the Chroma database of {dlg.DIALOG_NAME_ALFRED2} is read only."
)


# ==========================================================
# Sidebar: setting up the conversation
# ==========================================================

with st.sidebar:
    st.subheader("Setup")

    vorlage_name = st.selectbox("Template", options=list(dlg.VORLAGEN.keys()))
    vorlage = dlg.VORLAGEN[vorlage_name]

    thema = st.text_area(
        "Topic", value=vorlage["thema"], height=90,
        help="Leave empty for a free conversation without a given topic.",
    )

    st.divider()
    st.markdown(f"**{dlg.DIALOG_NAME_ALFRED2}** · here on the laptop")
    modell_a = st.selectbox("Model A", options=dlg.DIALOG_MODELLE,
                            index=dlg.DIALOG_MODELLE.index(dlg.DIALOG_MODELL_ALFRED2)
                            if dlg.DIALOG_MODELL_ALFRED2 in dlg.DIALOG_MODELLE else 0)
    rolle_a = st.text_area("Role A", value=vorlage["rolle_a"], height=110)
    mem_a = st.checkbox("uses its memory", value=True, key="mem_a")
    web_a = st.checkbox("may search the web", value=False, key="web_a")
    speichern_a = st.checkbox(
        "may learn and save to Chroma", value=False, key="save_a",
        help=f"Off by default. Switched on, {dlg.DIALOG_NAME_ALFRED2} also extracts "
             f"facts from the conversation and files them in your real database.",
    )
    if speichern_a:
        st.warning(
            f"{dlg.DIALOG_NAME_ALFRED2} writes into the **real Chroma database** in this run. "
            f"The entries are phrased as hearsay (“According to {dlg.DIALOG_NAME_LITE} …”) and "
            "carry `source: dialog:<id>` – in the Memory Viewer you can find them through the "
            "text search for “According to”.",
            icon="⚠️",
        )
    else:
        st.caption("Writes nothing into the Chroma database.")

    st.divider()
    st.markdown(f"**{dlg.DIALOG_NAME_LITE}** · on its own machine")
    pi_url = st.text_input("Address of the dialog server", value=dlg.DIALOG_PARTNER_URL)
    pi_token = st.text_input("Token", value=dlg.DIALOG_PARTNER_TOKEN, type="password",
                             help="Must match the token the partner server expects.")
    modell_b = st.selectbox(
        "Model B", options=["(whatever the partner has configured)"] + dlg.DIALOG_MODELLE,
    )
    rolle_b = st.text_area("Role B", value=vorlage["rolle_b"], height=110)
    mem_b = st.checkbox("uses its memory", value=True, key="mem_b")
    speichern_b = st.checkbox("may learn and save", value=True, key="save_b")
    web_b = st.checkbox("may search the web", value=False, key="web_b")

    st.divider()
    st.markdown("**Limits**")
    max_zuege = st.slider("Number of turns", min_value=2, max_value=40,
                          value=dlg.DIALOG_MAX_ZUEGE, step=1)
    kostendeckel = st.number_input("Cost cap in $", min_value=0.05, max_value=20.0,
                                   value=float(dlg.DIALOG_KOSTENDECKEL), step=0.25)
    beginner = st.radio("Who begins?",
                        options=[dlg.DIALOG_NAME_LITE, dlg.DIALOG_NAME_ALFRED2],
                        horizontal=True)

    st.divider()
    if st.button("🔌 Partner reachable?", use_container_width=True):
        probe = dlg.RemoteTeilnehmer(url=pi_url, token=pi_token)
        try:
            zustand = probe.status()
            st.success(
                f"{zustand.get('name')} answers · model `{zustand.get('model')}` · "
                f"{zustand.get('memories_inherited')} inherited and "
                f"{zustand.get('memories_own')} own memories"
            )
        except Exception as e:
            st.error(f"No answer from {pi_url}\n\n{e}")


# ==========================================================
# Controls
# ==========================================================

dialog = st.session_state.dialog

s1, s2, s3, s4 = st.columns([1, 1, 1, 3])

with s1:
    if st.button("▶️ Start", type="primary", use_container_width=True,
                 disabled=dialog is not None and st.session_state.laeuft):
        alfred2 = dlg.LokalerTeilnehmer(
            model=modell_a, use_memory=mem_a, web_search=web_a,
            save_memory=speichern_a,
        )
        lite = dlg.RemoteTeilnehmer(
            url=pi_url, token=pi_token,
            model="" if modell_b.startswith("(") else modell_b,
            use_memory=mem_b, save_memory=speichern_b, web_search=web_b,
        )
        st.session_state.dialog = dlg.Dialog(
            a=alfred2, b=lite, thema=thema,
            rolle_a=rolle_a, rolle_b=rolle_b,
            max_zuege=max_zuege, kostendeckel=kostendeckel,
            beginner=beginner,
        )
        st.session_state.laeuft = True
        st.session_state.pfad = None
        st.rerun()

with s2:
    if st.button("⏸️ Pause", use_container_width=True,
                 disabled=not st.session_state.laeuft):
        st.session_state.laeuft = False
        st.rerun()

with s3:
    if st.button("⏹️ Finish", use_container_width=True, disabled=dialog is None):
        if dialog:
            dialog.stoppe()
        st.session_state.laeuft = False
        st.rerun()

with s4:
    if dialog:
        offen = dialog.fertig()
        if offen:
            st.caption(f"Finished – {offen}")
        elif st.session_state.laeuft:
            st.caption(f"Running … turn {len(dialog.zuege) + 1} of {dialog.max_zuege}")
        else:
            st.caption("Paused – “Continue” picks it up again.")

if dialog is None:
    st.info(
        "Define the setup on the left and then start. Beforehand it is worth a click "
        "on “Partner reachable?” – without a running dialog server on the other end "
        f"{dlg.DIALOG_NAME_LITE} cannot answer."
    )
    st.stop()


# ==========================================================
# Key figures
# ==========================================================

k1, k2, k3, k4 = st.columns(4)
k1.metric("Turns", f"{len(dialog.zuege)} / {dialog.max_zuege}")
k2.metric("Cost", f"${dialog.kosten:.4f}",
          help=f"Cap: ${dialog.kostendeckel:.2f}")
k3.metric("Learned", sum(len(z.erinnerungen_gespeichert) for z in dialog.zuege),
          help=f"New memories that {dlg.DIALOG_NAME_LITE} has saved on its own machine.")
k4.metric("Conversation", dialog.konversations_id)

if dialog.kostendeckel:
    anteil = min(dialog.kosten / dialog.kostendeckel, 1.0)
    st.progress(anteil, text=f"Cost cap {anteil * 100:.0f} % used up")

st.divider()


# ==========================================================
# Course of the conversation
# ==========================================================

AVATARE = {dialog.a.name: "🤖", dialog.b.name: "🍓", USER_NAME: "🧑"}

for zug in dialog.zuege:
    with st.chat_message(zug.sprecher, avatar=AVATARE.get(zug.sprecher, "💬")):
        st.markdown(f"**{zug.sprecher}**")

        if zug.fehler:
            st.error(zug.fehler)
            continue

        st.write(zug.text)

        genutzt_gesamt = sum(len(v) for v in zug.erinnerungen_genutzt.values())
        gespeichert = len(zug.erinnerungen_gespeichert)

        fuss = f"`{zug.model}` · ${zug.kosten:.5f} · {zug.dauer:.1f}s"
        if zug.web_suchen:
            fuss += f" · {zug.web_suchen}× web search"
        st.caption(fuss)

        # The genuinely interesting part: what did this side have in mind, and
        # what did it take away? The conversation text alone shows only the
        # surface of the behaviour.
        # If the filter found almost nothing, the answer is in all likelihood
        # not backed by memories - and that is exactly when models tend to
        # invent something plausible. That belongs out in the open, not hidden
        # away in a collapsed box.
        if zug.erinnerungen_gesamt and genutzt_gesamt <= max(5, zug.erinnerungen_gesamt // 50):
            st.warning(
                f"Only {genutzt_gesamt} of {zug.erinnerungen_gesamt} memories "
                f"drawn on – this answer hardly rests on stored knowledge. "
                f"Read what is here especially critically.",
                icon="⚠️",
            )

        if genutzt_gesamt or gespeichert:
            titel = []
            if genutzt_gesamt:
                anteil = (f"{genutzt_gesamt} of {zug.erinnerungen_gesamt}"
                          if zug.erinnerungen_gesamt else str(genutzt_gesamt))
                titel.append(f"{anteil} memory/memories used")
            if gespeichert:
                titel.append(f"{gespeichert} newly saved")

            with st.expander("🧠 " + " · ".join(titel)):
                # Deliberately ONE markdown block per list instead of one call
                # per entry: with several hundred memories that would otherwise
                # be just as many Streamlit elements per turn - slow, and the
                # individual one-item lists render badly.
                for quelle, eintraege in zug.erinnerungen_genutzt.items():
                    if not eintraege:
                        continue
                    st.markdown(f"**Drawn on – {quelle} ({len(eintraege)}):**")
                    st.markdown("\n".join(f"- {m}" for m in eintraege if str(m).strip()))

                if gespeichert:
                    st.markdown("**Newly saved:**")
                    st.markdown("\n".join(
                        f"- `[{m.get('type')}]` {m.get('text')}"
                        for m in zug.erinnerungen_gespeichert
                    ))

# The observer's interjections appear in the transcript as well.
for eintrag in dialog.verlauf:
    if eintrag["sprecher"] == USER_NAME:
        with st.chat_message(USER_NAME, avatar="🧑"):
            st.write(eintrag["text"])


# ==========================================================
# Intervening
# ==========================================================

grund = dialog.fertig()

if not grund:
    e1, e2 = st.columns([3, 1])
    with e1:
        einwurf = st.text_input(
            "Interjection", placeholder="Throw in something that both of them read …",
            label_visibility="collapsed",
        )
    with e2:
        if st.button("Throw in", use_container_width=True, disabled=not einwurf.strip()):
            dialog.einwurf(einwurf.strip())
            st.rerun()

    if not st.session_state.laeuft:
        w1, w2 = st.columns([1, 4])
        with w1:
            if st.button("▶️ Continue", type="primary", use_container_width=True):
                st.session_state.laeuft = True
                st.rerun()
        with w2:
            if st.button("Just one turn", use_container_width=True):
                with st.spinner(f"{dialog.naechster_sprecher.name} is thinking …"):
                    dialog.naechster_zug()
                st.rerun()

else:
    st.success(f"Conversation finished – {grund}")

    b1, b2 = st.columns([1, 3])
    with b1:
        if st.button("💾 Save transcript", use_container_width=True):
            st.session_state.pfad = dialog.speichere()
    with b2:
        if st.session_state.pfad:
            st.caption(f"Saved: `{st.session_state.pfad}`")

    if st.button("🔄 New conversation"):
        st.session_state.dialog = None
        st.session_state.laeuft = False
        st.session_state.pfad = None
        st.rerun()


# ==========================================================
# One turn per run
# ==========================================================
# This deliberately sits right at the end: first the transcript so far is
# rendered, then the next turn is fetched and a rerun is triggered. That way
# you see every contribution appear instead of everything at once at the end.

if st.session_state.laeuft and not dialog.fertig():
    with st.spinner(f"{dialog.naechster_sprecher.name} is thinking …"):
        dialog.naechster_zug()
    st.rerun()

if st.session_state.laeuft and dialog.fertig():
    st.session_state.laeuft = False
    # Save the transcript automatically so that a run is never lost.
    if not st.session_state.pfad:
        st.session_state.pfad = dialog.speichere()
    st.rerun()
