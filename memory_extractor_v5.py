"""
memory_extractor_v5.py

Alfred 2.0 - Memory Extractor v5

What's new:
- Atomic memories
- Typing (type)
- Confidence + reason
- Strict information extractor
"""

import json
import re
import time

import requests

try:
    from config import OLLAMA_HOST, MEMORY_EXTRACTOR_MODEL, USER_NAME, OLLAMA_NUM_THREAD, ASSISTANT_LANGUAGE_NAME
except Exception:
    OLLAMA_HOST = "http://localhost:11434"
    MEMORY_EXTRACTOR_MODEL = "llama3.2"
    USER_NAME = "Jonas"
    OLLAMA_NUM_THREAD = None
    ASSISTANT_LANGUAGE_NAME = "English"


ALLOWED_TYPES = {
    "profession", "workplace", "family", "device", "interest",
    "hobby", "project", "goal", "skill", "preference", "other",
}


class MemoryExtractor:

    def __init__(self,
                 model=MEMORY_EXTRACTOR_MODEL,
                 host=OLLAMA_HOST,
                 temperature=0.0,
                 retries=3,
                 debug=False,
                 keep_alive="30m",
                 num_thread=OLLAMA_NUM_THREAD,
                 retry_pause=1.5):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.retries = retries
        self.debug = debug
        self.keep_alive = keep_alive
        self.num_thread = num_thread
        # Named in the "unreachable" warning. The cloud subclass
        # overrides it - being told to check Ollama when you never
        # installed it is worse than no message at all.
        self.backend_name = "Ollama"
        # True when the last extract() could not reach the backend at
        # all. Callers need this to tell "the model found nothing"
        # from "nobody answered" - the two look identical otherwise,
        # because both end up as an empty list.
        self.last_call_failed = False
        self.retry_pause = retry_pause

    def _build_prompt(self, conversation, user_name=None):
        # Who the conversation is about. Without this the model has no way
        # of knowing whose facts it is writing down: a sentence like "my
        # hobby is testing" names no subject, and the rules below demand
        # one. Before this was passed in, the model either dropped such
        # facts entirely or - worse - borrowed a name from the EXAMPLES
        # section further down and filed "Jonas's hobby is testing."
        #
        # It is a parameter rather than an attribute on purpose: one
        # extractor instance serves every user of an installation, so the
        # name has to come with the call, not with the object.
        person = (user_name or USER_NAME or "").strip() or "the user"

        return f"""
### WHO IS SPEAKING

The person speaking in the CONVERSATION is called {person}. Everything
they say about themselves is about {person}, including sentences in the
first person ("I", "my", "me"). Write every memory with {person} as the
explicit subject, for example "{person} likes pizza."

NEVER use a name that does not appear either here or in the CONVERSATION
itself. The names in the EXAMPLES section below are invented and serve
only to show the JSON format - they must never turn up in your answer.

### TASK

You are exclusively an information extractor.

Extract permanent facts from the conversation.

### RULES

- MOST IMPORTANT RULE, TAKES PRECEDENCE OVER ALL OTHERS EXCEPT THE OPT-IN RULE DIRECTLY BELOW: If the user explicitly says in the CONVERSATION that you should not/must not/may not/need not remember something (e.g. "you don't have to remember that", "you shouldn't store that", "you must not remember that", "you don't need to save that", "forget that again", "that's not important for you"), then extract NOTHING AT ALL from the ENTIRE conversation - not even the part before this instruction. When in doubt: better to store nothing than a memory that the user explicitly did not want stored.
- OPT-IN RULE, TAKES PRECEDENCE OVER ALL OTHER RULES (including the rules on small talk/temporary things/questions further below): If the user explicitly says in the CONVERSATION that you should/must/may remember something (e.g. "you have to remember that", "you should remember that", "you're welcome to remember that", "please remember that", "you should know and store that"), then ALWAYS extract the mentioned information as a memory - even if it would otherwise count as small talk, temporary or unimportant. If no other type fits, use "other". If the CONVERSATION contains both instructions at the same time (contradictory), the do-not-remember rule takes precedence.
- Only take over explicitly mentioned information.
- Do not use prior knowledge.
- Do not invent anything.
- Do not interpret anything.
- Do not infer anything.
- MOST IMPORTANT RULE AGAINST INFERENCES: For EVERY fact you want to extract, check: "Is this stated in the CONVERSATION WORD FOR WORD or as an immediate paraphrase?" If the answer is no, or if you are not sure, do NOT extract the fact. A statement about a PROPERTY of an object (e.g. "My coffee is hot.") is NOT a preference or any other piece of information about the user - it does NOT follow that the user likes coffee.
- NEVER write phrasings like "presumably", "probably", "not explicitly mentioned, but", "could mean" in the memory text. If you notice that you are about to write something like that, it is a sign that you are currently inferring instead of extracting - in that case leave the fact out completely.
- Write all memories exclusively in {ASSISTANT_LANGUAGE_NAME}.
- Split complex statements into atomic memories.
- IMPORTANT RULE FOR ENUMERATIONS: Every single atomic memory MUST be a complete sentence with an explicitly named subject (who?) AND a verb - even if an enumeration with a shared subject occurs in the CONVERSATION (e.g. "Lena likes LEGO, Batman and Star Wars."). Repeat the subject and a fitting verb for EVERY single element of the enumeration, not just for the first one. A memory text that consists only of a single word or a bare phrase without subject and verb (e.g. "LEGO", "Batman", "Champions League", "electronic music") is INVALID and must NEVER be produced - not even as part of an enumeration.
- SELF-CHECK BEFORE OUTPUT: Read every memory text you want to produce on its own - without knowing the CONVERSATION that came before it. If, while doing so, you CANNOT immediately tell (a) WHO it is about and WHAT is being said about that person, OR (b) WHAT the statement refers to (e.g. place, trip, project, event), in case the CONVERSATION established such a reference earlier and the sentence would otherwise be incomprehensible on its own or applicable to anything at random, then add what is missing from the CONVERSATION (without inventing anything new) or leave the fact out completely if the REFERENCE cannot be clarified unambiguously. The SUBJECT is never unclear in this sense: it is always {person}, named at the top - a first-person sentence is about {person}, so never drop a fact merely because the CONVERSATION says "I" instead of a name. Example of a missing reference: "There are areas there that are 200 meters above sea level." is useless on its own if the CONVERSATION had previously been about a trip to Crete - in that case the memory instead reads e.g. "{person} says there are viewpoints on Crete around 200 meters above sea level."
- Return JSON exclusively.
- "type" MUST be exactly one of the values from MEMORY TYPES. NEVER invent your own types (e.g. "greeting" or "weather" are FORBIDDEN).
- Greetings, farewells, pleasantries ("Thanks!", "How are you?", "See you tomorrow."), weather information and purely temporary states (tiredness, current mood, "today", "right now", "now") are NOT permanent facts. If only such things occur in the CONVERSATION: return {{"memories":[]}}.
- Questions are NOT facts, even if they touch on a topic such as profession, family or the like (e.g. "Where do I work?", "Do you know who I am?"). NEVER extract a memory from a question - not even a fragment of the text of the question itself. Only if the user themselves STATES a fact in the CONVERSATION (no question mark, no interrogative form) are you allowed to extract.
- Everyday preferences (likes X, enjoys eating/drinking X, prefers X over Y) ARE permanent facts and should be stored as type "preference" - that makes the memories about the user more interesting. But: a statement about a property of a specific object at this moment (e.g. "My coffee is hot.") is NOT a preference.
- A decision that has not yet been made, or current deliberating/wavering ("I'm torn between X and Y", "I'm undecided whether..."), is NOT a permanent preference and NOT a fact - that is a temporary snapshot. Return {{"memories":[]}} for that.
- "interest" is for topics/subject areas the user is interested in (e.g. "is interested in astrophotography"). "preference" is for concrete everyday preferences such as food/drink (e.g. "likes pizza", "enjoys drinking coffee").
- The examples below serve ONLY to illustrate the JSON format. NEVER copy the example text into your answer. Extract exclusively from the text under "### CONVERSATION" further below - not from the examples.

### MEMORY TYPES

profession
workplace
family
device
interest
hobby
project
goal
skill
preference
other

### EXAMPLES

Conversation:
Jonas works as a Java developer at Stadtwerke Musterstadt.

Answer:

{{
  "memories":[
    {{
      "type":"profession",
      "text":"Jonas works as a Java developer.",
      "confidence":1.0,
      "reason":"Long-term information"
    }},
    {{
      "type":"workplace",
      "text":"Jonas works at Stadtwerke Musterstadt.",
      "confidence":1.0,
      "reason":"Long-term information"
    }}
  ]
}}

Conversation:
Jonas likes pizza.

Answer:

{{
  "memories":[
    {{
      "type":"preference",
      "text":"Jonas likes pizza.",
      "confidence":1.0,
      "reason":"Everyday preference"
    }}
  ]
}}

Conversation:
Lena likes LEGO, Batman and Star Wars.

Answer:

{{
  "memories":[
    {{
      "type":"hobby",
      "text":"Lena likes LEGO.",
      "confidence":1.0,
      "reason":"Everyday preference"
    }},
    {{
      "type":"interest",
      "text":"Lena likes Batman.",
      "confidence":1.0,
      "reason":"Everyday preference"
    }},
    {{
      "type":"interest",
      "text":"Lena likes Star Wars.",
      "confidence":1.0,
      "reason":"Everyday preference"
    }}
  ]
}}

Conversation:
We're thinking about the Kreta vacation in September. There are areas there that you can drive to, maybe 200 meters above sea level, with a great view.

Answer:

{{
  "memories":[
    {{
      "type":"other",
      "text":"Jonas is planning a vacation on Kreta in September.",
      "confidence":1.0,
      "reason":"Long-term information"
    }},
    {{
      "type":"other",
      "text":"According to Jonas, there are viewpoints on Kreta around 200 meters above sea level with a great view.",
      "confidence":1.0,
      "reason":"Reference taken over from the conversation so that the fact remains understandable on its own"
    }}
  ]
}}

Conversation:
Hello Alfred!

Answer:

{{"memories":[]}}

Conversation:
How are you?

Answer:

{{"memories":[]}}

Conversation:
Thanks!

Answer:

{{"memories":[]}}

Conversation:
It's raining today.

Answer:

{{"memories":[]}}

Conversation:
I'm tired right now.

Answer:

{{"memories":[]}}

Conversation:
I'm on vacation today.

Answer:

{{"memories":[]}}

Conversation:
My coffee is hot.

Answer:

{{"memories":[]}}

Conversation:
Jonas lives with his family in Musterstadt.

Answer:

{{
  "memories":[
    {{
      "type":"family",
      "text":"Jonas lives with his family in Musterstadt.",
      "confidence":1.0,
      "reason":"Long-term information"
    }}
  ]
}}

Conversation:
I'd imagine that you know where I work, where do I work?

Answer:

{{"memories":[]}}

Conversation:
I'm currently torn between pizza and sushi for dinner.

Answer:

{{"memories":[]}}

Conversation:
By the way, I enjoy making cappuccino, but you don't have to remember that.

Answer:

{{"memories":[]}}

Conversation:
I'm in a really bad mood today, but you're welcome to remember that.

Answer:

{{
  "memories":[
    {{
      "type":"other",
      "text":"Jonas was in a bad mood.",
      "confidence":1.0,
      "reason":"User explicitly wanted this stored"
    }}
  ]
}}

### CONVERSATION

{conversation}

### JSON
"""

    def _call(self, prompt):
        options = {"temperature": self.temperature}
        if self.num_thread:
            options["num_thread"] = self.num_thread

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": options,
            "keep_alive": self.keep_alive,
        }
        r = requests.post(self.host + "/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def extract(self, conversation, user_name=None):
        """Extracts memories from one conversation turn.

        user_name is the person the conversation is about. Pass it
        whenever several people share one installation - the chat does
        this with the name typed on the start screen. Left out, it falls
        back to USER_NAME from config.py.
        """
        prompt = self._build_prompt(conversation, user_name)
        self.last_call_failed = False

        if self.debug:
            print("="*70)
            print("PROMPT")
            print("="*70)
            print(prompt)

        network_errors = 0

        for attempt in range(1, self.retries + 1):
            if self.debug:
                print(f"[DEBUG] Attempt {attempt}")

            try:
                raw = self._call(prompt)
            except requests.exceptions.RequestException as e:
                network_errors += 1
                if self.debug:
                    print(f"[DEBUG] Network error on attempt {attempt}: {e}")
                # Wait before trying again. Retrying a rate limit or a brief
                # outage in the same instant simply fails again; a second or
                # two is usually all it takes.
                if attempt < self.retries:
                    time.sleep(self.retry_pause * attempt)
                continue

            if self.debug:
                print("="*70)
                print("RAW")
                print("="*70)
                print(raw)

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                continue

            try:
                obj = json.loads(match.group(0))
            except Exception:
                continue

            memories = obj.get("memories", [])

            cleaned = []
            seen = set()

            for m in memories:
                if not isinstance(m, dict):
                    continue

                text = m.get("text", "").strip()

                if not text or text in seen:
                    continue

                mem_type = m.get("type", "other")

                if mem_type not in ALLOWED_TYPES:
                    if self.debug:
                        print(f"[DEBUG] Discarded (invalid type '{mem_type}'): {text}")
                    continue

                if text.endswith("?"):
                    if self.debug:
                        print(f"[DEBUG] Discarded (looks like a question): {text}")
                    continue

                seen.add(text)

                cleaned.append({
                    "type": mem_type,
                    "text": text,
                    "confidence": float(m.get("confidence", 1.0)),
                    "reason": m.get("reason", "Long-term information")
                })

            return cleaned

        if network_errors == self.retries:
            self.last_call_failed = True
            print(
                f"⚠️  WARNING: {self.backend_name} was unreachable on ALL "
                f"{self.retries} attempts (timeout/network error). The result "
                f"'no memory' for this conversation is NOT meaningful - "
                f"check {self.backend_name}!"
            )

        return []


if __name__ == "__main__":

    extractor = MemoryExtractor(debug=True)

    while True:

        print()
        conversation = input("Conversation (empty to quit): ")

        if not conversation:
            break

        memories = extractor.extract(conversation)

        print("\nMemories:\n")

        if not memories:
            print("No memories found.")
            continue

        for i, mem in enumerate(memories, 1):
            print(f"{i:02d}. [{mem['type']}] {mem['text']}")
            print(f"    Confidence: {mem['confidence']}")
            print(f"    Reason    : {mem['reason']}")
            print()
