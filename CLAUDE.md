# Aible — Project Instructions

## What this is

*Agentic Systems: Patterns and Teardowns* — a short book on designing multi-agent AI
systems, sitting on top of a ~45-page AI reference library inherited from v1 of this
site. Docusaurus 3.10, deployed to GitHub Pages at `chintan-dshel.github.io/aible`.

Work happens on branch `aible-v2`. Never commit to `main`.
`V2_PLAN.md` is the source of truth for structure and phase status — read it before
starting work in a new session. Do not trust a session summary over that file.

## Model routing policy

Claude cannot change the main session model; that is the user's `/model` setting.
What Claude controls is which model each **subagent** runs on. Route deliberately —
the user pays for Opus tokens.

**Send to Sonnet** (`model: "sonnet"` on the Agent tool):
- Generating files from a specified template (stub pages, boilerplate)
- Mechanical sweeps: find-and-replace across many files, link audits, frontmatter fixes
- Inventory and counting tasks: "list every page with X", grep-shaped questions
- Verification runs where the pass/fail criterion is already written down
- Formatting and lint passes against an explicit rule set

**Keep on Opus** (default, or `model: "opus"`):
- Prose voice work — writing or rewriting chapters, anything where tone is the deliverable
- Reading a codebase to produce a teardown; any claim that must be verified against source
- Architecture and information-architecture decisions
- Reviews where the reviewer must decide what matters, not just apply a checklist

**Rule of thumb:** if the task has a written spec and a checkable output, it is Sonnet
work. If the task *produces* the spec or requires taste, it is Opus work.

**Also reduce main-session burn:**
- Batch edits, then build once. Do not rebuild after each file.
- Do not re-read a file already in context.
- Screenshot-verify once per batch, not per change.
- Prefer one targeted probe over broad exploratory reads.

## Conventions

**Chapter shape (patterns).** Six sections, in this order: the problem in one diagram /
the pattern / decision rules (closing on one bolded portable test) / failure modes /
where it shows up in the teardowns / reference material.

**Teardown shape.** Eight sections: what it does / architecture / control flow / data
flow / design decisions and trade-offs / strengths / weaknesses / how it could be
attacked / what I'd change. Every claim read from source at a named commit; anything
inferred is tagged `[VERIFY]`.

**Voice.** Written by a human for humans. Specifics over abstractions: real commit
hashes, dollar figures, dates, file paths. First person where the author made the
decision — "I ripped it out," not "it was removed." No paragraph ending on an
aphorism. No "X is not Y, it is Z" unless two artifacts literally contradict. Never
announce a count before a list. Do not restate the figure caption in the prose under
it. See `docs/STYLE.md`.

**Diagrams.** Mermaid, one idea each, max ~12 nodes. Node classes `:::orch :::agent
:::gate :::store :::human :::external :::fail` — colours live in `src/css/custom.css`,
never as per-diagram `classDef`. Every diagram carries `accTitle`/`accDescr` and a
caption `<p className="fig-caption"><strong>Figure N.n</strong> — …</p>`.

**Never restyle Mermaid label typography in CSS.** Mermaid measures labels and fixes
the SVG box size at render time; changing font-size or padding afterward reflows the
text outside its box and clips it. Set label sizing in `docusaurus.config.ts`
(`mermaid.options`) so measurement and render agree. Colour-only CSS overrides are safe.

**Two or more disconnected subgraphs in one flowchart** have no guaranteed relative
order — dagre may reverse them. Add an invisible ordering edge (`A --- B` plus
`linkStyle N stroke:none`).

## Verifying

Build with `npm run build` — it reports broken links; treat any as a failure.
For visual checks, serve the build and screenshot in headless Chrome over the DevTools
protocol (Chrome must be launched from the shell, not spawned from Node, on this
machine). Confirm the rendered SVG count matches the number of diagrams on the page —
a Mermaid block that fails to parse disappears silently with no build error.

## Second Brain

Session lessons go to `D:\AI_Projects\second-brain\wiki\Aible-v2-Restructure-Lessons.md`
at the end of each session, plus an entry in `wiki/log.md`.
