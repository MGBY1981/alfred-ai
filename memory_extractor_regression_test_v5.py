"""
memory_extractor_regression_test_v5.py

Large regression test with verbose output.

- 43 test cases
- PASS / WARNING / TYPE FAIL / FAIL
- conversation, expected, received per test
- statistics per category

The test cases themselves are deliberately kept in German: this suite
measures the quality of the GERMAN memory extraction. Everything the
suite needs - including the scoring helpers - lives in this one file,
so it runs on its own.
"""

from collections import defaultdict

from config_guard import require_config
require_config()   # readable message if config.py is missing

import config
from memory_extractor_v5 import MemoryExtractor

import memory_extractor_v5
# The 43 test cases below are written in German and expect German
# memories back, so pin the extractor's output language regardless of
# what config.py says. The score this produces therefore always
# describes GERMAN extraction, whatever language the assistant runs in.
memory_extractor_v5.ASSISTANT_LANGUAGE_NAME = "German"

# The expected results name "Jonas", so the speaker has to be Jonas no
# matter whose name stands in config.py. Pinned here rather than left
# to chance: without it the score would depend on the machine the test
# runs on.
TEST_USER = "Jonas"


# ==========================================================
# Scoring
# ==========================================================
# These two helpers used to live in an older revision of this test.
# They are inlined here so the regression test is a single
# self-contained file.


def norm(text):
    """Comparison form for scoring: case and surrounding whitespace are
    irrelevant, a trailing full stop is not a difference."""
    return (text or "").strip().lower().rstrip(".")


def score_case(expected, actual):
    """Compares the memories the extractor produced against the expected
    ones and returns one of four verdicts.

    PASS       everything matched exactly and nothing extra was produced
    WARNING    content matches, but not word for word, or there is an
               extra memory on top
    TYPE FAIL  the text is right but the assigned type is wrong
    FAIL       an expected memory is missing entirely, or something was
               produced where nothing should have been
    """
    if not expected and not actual:
        return "PASS"

    if not expected and actual:
        return "FAIL"

    if expected and not actual:
        return "FAIL"

    exact = True
    type_problem = False

    for etype, etext in expected:
        found = False
        for mem in actual:
            atype = mem.get("type", "")
            atext = mem.get("text", "")

            if norm(etext) == norm(atext):
                found = True
                if etype != atype:
                    type_problem = True
                break

            # A substring counts as acceptable in substance.
            if norm(etext) in norm(atext) or norm(atext) in norm(etext):
                found = True
                exact = False
                if etype != atype:
                    type_problem = True
                break

        if not found:
            return "FAIL"

    if type_problem:
        return "TYPE FAIL"

    if exact and len(expected) == len(actual):
        return "PASS"

    return "WARNING"


TESTS = [

# Small talk
{"name":"Smalltalk 1","conversation":"Hallo Alfred!","expected":[]},
{"name":"Smalltalk 2","conversation":"Guten Morgen.","expected":[]},
{"name":"Smalltalk 3","conversation":"Wie geht's?","expected":[]},
{"name":"Smalltalk 4","conversation":"Danke!","expected":[]},
{"name":"Smalltalk 5","conversation":"Bis morgen.","expected":[]},

# Temporary
{"name":"Temporär 1","conversation":"Heute regnet es.","expected":[]},
{"name":"Temporär 2","conversation":"Ich bin gerade müde.","expected":[]},
{"name":"Temporär 3","conversation":"Es ist kalt.","expected":[]},
{"name":"Temporär 4","conversation":"Mein Kaffee ist heiß.","expected":[]},
{"name":"Temporär 5","conversation":"Ich habe heute Urlaub.","expected":[]},
{"name":"Temporär 6","conversation":"Ich überlege gerade zwischen Pizza und Sushi zum Abendessen.","expected":[]},
{"name":"Temporär 7","conversation":"Übrigens mach ich gerne Cappuccino, aber das musst du dir nicht merken.","expected":[]},
{"name":"Temporär 8","conversation":"Ich habe heute richtig schlechte Laune, das darfst du dir aber gerne merken.","expected":[("other","Jonas hatte schlechte Laune.")]},

# Profession
{"name":"Beruf 1","conversation":"Jonas arbeitet als Java-Entwickler.","expected":[("profession","Jonas arbeitet als Java-Entwickler.")]},
{"name":"Beruf 2","conversation":"Jonas arbeitet in der IT.","expected":[("profession","Jonas arbeitet in der IT.")]},
{"name":"Beruf 3","conversation":"Jonas ist Elektronikingenieur.","expected":[("profession","Jonas ist Elektronikingenieur.")]},
{"name":"Beruf 4","conversation":"Jonas arbeitet bei den Stadtwerken Musterstadt.","expected":[("workplace","Jonas arbeitet bei den Stadtwerken Musterstadt.")]},
{"name":"Beruf 5","conversation":"Jonas arbeitet als Java-Entwickler bei den Stadtwerken Musterstadt.","expected":[("profession","Jonas arbeitet als Java-Entwickler."),("workplace","Jonas arbeitet bei den Stadtwerken Musterstadt.")]},

# Devices
{"name":"Gerät 1","conversation":"Jonas besitzt eine Canon EOS 600D.","expected":[("device","Jonas besitzt eine Canon EOS 600D.")]},
{"name":"Gerät 2","conversation":"Jonas besitzt eine Fender Stratocaster.","expected":[("device","Jonas besitzt eine Fender Stratocaster.")]},
{"name":"Gerät 3","conversation":"Jonas hat einen Raspberry Pi 5.","expected":[("device","Jonas hat einen Raspberry Pi 5.")]},
{"name":"Gerät 4","conversation":"Jonas besitzt ein iPhone.","expected":[("device","Jonas besitzt ein iPhone.")]},
{"name":"Gerät 5","conversation":"Jonas besitzt einen Echo Dot.","expected":[("device","Jonas besitzt einen Echo Dot.")]},

# Interests
{"name":"Interesse 1","conversation":"Jonas interessiert sich für Astrofotografie.","expected":[("interest","Jonas interessiert sich für Astrofotografie.")]},
{"name":"Interesse 2","conversation":"Jonas interessiert sich für KI.","expected":[("interest","Jonas interessiert sich für KI.")]},
{"name":"Interesse 3","conversation":"Jonas interessiert sich für Gitarren.","expected":[("interest","Jonas interessiert sich für Gitarren.")]},
{"name":"Interesse 4","conversation":"Jonas interessiert sich für Linux.","expected":[("interest","Jonas interessiert sich für Linux.")]},
{"name":"Interesse 5","conversation":"Jonas interessiert sich für Raumfahrt.","expected":[("interest","Jonas interessiert sich für Raumfahrt.")]},

# Projects
{"name":"Projekt 1","conversation":"Jonas entwickelt Alfred 2.0.","expected":[("project","Jonas entwickelt Alfred 2.0.")]},
{"name":"Projekt 2","conversation":"Jonas baut einen KI-Agenten.","expected":[("project","Jonas baut einen KI-Agenten.")]},
{"name":"Projekt 3","conversation":"Jonas programmiert einen Chatbot.","expected":[("project","Jonas programmiert einen Chatbot.")]},
{"name":"Projekt 4","conversation":"Jonas entwickelt ein Ansible-Projekt.","expected":[("project","Jonas entwickelt ein Ansible-Projekt.")]},
{"name":"Projekt 5","conversation":"Jonas arbeitet an einem YouTube-Video.","expected":[("project","Jonas arbeitet an einem YouTube-Video.")]},

# Family
{"name":"Familie 1","conversation":"Jonas hat zwei Söhne.","expected":[("family","Jonas hat zwei Söhne.")]},
{"name":"Familie 2","conversation":"Jonas ist verheiratet.","expected":[("family","Jonas ist verheiratet.")]},
{"name":"Familie 3","conversation":"Jonas lebt mit seiner Familie in Musterstadt.","expected":[("family","Jonas lebt mit seiner Familie in Musterstadt.")]},
{"name":"Familie 4","conversation":"Jonas hat einen Bruder.","expected":[("family","Jonas hat einen Bruder.")]},
{"name":"Familie 5","conversation":"Jonas wohnt mit seiner Frau zusammen.","expected":[("family","Jonas wohnt mit seiner Frau zusammen.")]},

# Edge cases
{"name":"Grenzfall 1","conversation":"Jonas mag Pizza.","expected":[("preference","Jonas mag Pizza.")]},
{"name":"Grenzfall 2","conversation":"Jonas trinkt gerne Kaffee.","expected":[("preference","Jonas trinkt gerne Kaffee.")]},
{"name":"Grenzfall 3","conversation":"Jonas liest gerade ein Buch.","expected":[]},
{"name":"Grenzfall 4","conversation":"Jonas spielt seit Jahren Gitarre.","expected":[("hobby","Jonas spielt seit Jahren Gitarre.")]},
{"name":"Grenzfall 5","conversation":"Jonas möchte Spanisch lernen.","expected":[("goal","Jonas möchte Spanisch lernen.")]}
]

# Keys are the verdicts returned by score_case() and must stay as they are;
# only the explanatory texts are translated.
REASONS={
    "ERROR": "The backend could not be reached - this says nothing about quality.",
"PASS":"Result matches the expectation.",
"WARNING":"Content largely matches, or an additional memory was detected.",
"TYPE FAIL":"Information was detected, but the type does not match.",
"FAIL":"Hallucination, irrelevant or missing memory."
}

def cat(name): return name.split()[0]

def show_expected(exp):
    if not exp: return "  (no memories)"
    return "\n".join(f"  - [{t}] {txt}" for t,txt in exp)

def show_actual(act):
    if not act: return "  (no memories)"
    return "\n".join(f"  - [{m.get('type')}] {m.get('text')}" for m in act)

# Which backend answers the 43 cases. Until now this was hard-wired to
# the local Ollama class, no matter what config.py said - so the score
# described llama3.1:8b even for someone whose chat runs on a cloud
# model. It follows the configuration now, and the model it actually
# used is printed with the result: a quality score without the model
# beside it says very little.
def _backend():
    modus = getattr(config, "MEMORY_EXTRACTOR_BACKEND", "openrouter")
    if modus == "openrouter":
        from memory_extractor_cloud import CloudMemoryExtractor
        e = CloudMemoryExtractor()
        return e, "OpenRouter", e.model
    if modus == "ollama":
        e = MemoryExtractor(
            model=getattr(config, "MEMORY_EXTRACTOR_MODEL", "llama3.1:8b"),
            host=getattr(config, "OLLAMA_HOST", "http://localhost:11434"),
            num_thread=getattr(config, "OLLAMA_NUM_THREAD", None),
            debug=False,
        )
        return e, "Ollama (local)", e.model
    raise SystemExit(
        "MEMORY_EXTRACTOR_BACKEND is 'off' - set it to 'openrouter' or "
        "'ollama' in config.py, otherwise there is nothing to measure."
    )


ext, BACKEND_NAME, BACKEND_MODEL = _backend()
stats=defaultdict(int)
cats=defaultdict(lambda:defaultdict(int))

print("="*80)
print(" Alfred Memory Extractor Regression Test v5 (43 tests)")
print("="*80)
print(f" Backend : {BACKEND_NAME}")
print(f" Model   : {BACKEND_MODEL}")
print(f" Language: German (pinned)   Speaker: {TEST_USER} (pinned)")
print("="*80)

for i,test in enumerate(TESTS,1):
    res=ext.extract(test["conversation"], user_name=TEST_USER)
    # A call that never reached the backend returns an empty list, exactly
    # like "the model decided there was nothing to store". Scoring those as
    # FAIL would blame the model for the network: a flaky connection would
    # quietly drag the quality score down and look like a regression.
    if getattr(ext, "last_call_failed", False):
        verdict = "ERROR"
    else:
        verdict=score_case(test["expected"],res)
    stats[verdict]+=1
    cats[cat(test["name"])][verdict]+=1

    print("\n"+"="*80)
    print(f"TEST {i}: {test['name']}")
    print("="*80)
    print("\nConversation\n"+"-"*80)
    print(test["conversation"])
    print("\nExpected\n"+"-"*80)
    print(show_expected(test["expected"]))
    print("\nReceived\n"+"-"*80)
    print(show_actual(res))
    print("\nVerdict\n"+"-"*80)
    print(verdict)
    print(REASONS[verdict])

print("\n"+"="*80)
print("Statistics by category")
print("="*80)
for c in sorted(cats):
    s=cats[c]
    print(f"\n{c}")
    for k in ["PASS","WARNING","TYPE FAIL","FAIL","ERROR"]:
        print(f"  {k:<10}: {s[k]}")

total=len(TESTS)
# Cases the backend never answered say nothing about quality, so they are
# left out of the score instead of counting against it.
gewertet = total - stats["ERROR"]
score = ((stats["PASS"] + 0.75*stats["WARNING"] + 0.4*stats["TYPE FAIL"])
         / gewertet * 100) if gewertet else 0.0

print("\n"+"="*80)
print("Overall result")
print("="*80)
for k in ["PASS","WARNING","TYPE FAIL","FAIL"]:
    print(f"{k:<10}: {stats[k]}")
if stats["ERROR"]:
    print(f"{'ERROR':<10}: {stats['ERROR']}   (backend unreachable - not counted)")
print(f"Tests        : {total}")
print(f"Scored       : {gewertet}")
print(f"Quality score: {score:.1f}%")
print(f"measured with: {BACKEND_NAME}, {BACKEND_MODEL}, German")

if stats["ERROR"]:
    anteil = stats["ERROR"] / total * 100
    print(f"\n⚠️  {stats['ERROR']} of {total} cases ({anteil:.0f}%) never reached "
          f"{BACKEND_NAME}.")
    if anteil > 10:
        print("   That is too many to call this a measurement. Run it again\n"
              "   when the connection is steady before quoting the number.")
    else:
        print("   They are excluded from the score above.")

print("\nQuote this number only together with the model beside it - a\n"
      "different model gives a different score on the same 43 cases.")
