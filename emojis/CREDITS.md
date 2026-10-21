# About these images

The 18 PNG files in this folder were generated with an AI image
generator (ChatGPT / DALL·E) by the author of this project.

**You may use them for anything, without restriction and without
attribution** - inside this project, in a fork of it, or somewhere else
entirely. No permission needs to be asked for and no credit needs to be
given.

Two notes, for the curious rather than the cautious:

- Purely prompt-generated images very likely carry no copyright in the
  first place. German courts have held that AI output is only protected
  where human creative shaping is recognisable in the specific image,
  and that prompting alone is not enough (OLG Düsseldorf, 2 April 2026,
  I-20 W 2/26; the Munich and Frankfurt courts have reached similar
  conclusions). So this is less a licence grant than a statement that
  nobody is going to come after you for these files.
- The code in this repository is under the MIT licence (see ../LICENSE).
  These images are not code, which is why they get their own note.

## Replacing them

The images are decorative - Alfred works without them. He writes
`[EMOJI:robot_happy]` into his answer and the chat client turns that
into the matching PNG; with the folder missing, the emoji list in the
system prompt is simply empty and he stops using them.

If you would rather not ship binary image files, each chat client reads
the folder in three places (`get_available_emojis`,
`render_message_with_emojis`, and the emoji list inside the system
prompt), so swapping in Unicode characters means touching those three
spots in `alfred.py` and the same three in `alfred_ollama.py`.

Or just drop your own PNGs in here: the file name without the extension
is the emoji name, and it reaches Alfred's system prompt by itself.

