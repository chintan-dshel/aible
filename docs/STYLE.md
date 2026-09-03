# Style guide

This file is the checkable criterion behind two review passes this project runs:
a voice pass (does this read as written by a human for humans) and an
accessibility pass (can a reader with no coding background follow it). Both are
referenced from `CLAUDE.md`. Treat every rule here as testable — if you can't
point at a sentence and say pass or fail, the rule isn't specific enough yet.

## Voice

Written by a human for humans. Established during the Part 1 voice rewrite
(2026-09-02) and applied to every chapter since.

1. **No paragraph ends on an aphorism.** Cover the last sentence with your
   thumb. If the paragraph still works, the closer was decoration — cut it.
2. **Budget one "X is not Y, it is Z" per chapter**, and only for a literal
   contradiction between two things (two artifacts disagreeing, not a
   rhetorical flourish).
3. **Never announce a count before a list.** Not "Figure 2.1 shows three
   things going wrong." Start with the first thing; let the reader count.
4. **Delete "Figure N.N shows/is/answers" sentences.** The caption already
   said it. Start the prose at the substance.
5. **Name who made the decision — usually "I."** Not "the gate was removed."
   "I ripped it out." Passive voice erases the person, and this book is
   written by a person who made real decisions.
6. **Claims about practice carry a number, a date, a file path, or a commit
   hash — or they get cut or explicitly hedged.** No invented precision
   ("within a week") standing in for a real measurement you don't have.
7. **Vary the shape of parallel entries.** Six failure modes in a row, each
   built as symptom-symptom-`Fix:`, reads as a generated table. At least one
   entry per section should break the pattern.
8. **No sentence appears in two chapters.** Each "Where it shows up in the
   teardowns" section is written from that teardown's actual content, not
   filled from a template.
9. **A "My take" callout contains a first-person verb and a fact only the
   author could know.** If you can `grep` the block for `\bI\b` and get zero
   hits, it isn't a take, it's the body restated in third person.
10. **Tell the anecdote before the abstraction, not after in a callout.** If
    you have a real story, it opens the section — it doesn't get relegated to
    a box at the bottom.

## Accessibility

Established 2026-09-02, in response to the author's read: teardowns and parts
of the pattern chapters were too technical for a non-coder audience. The goal
is genuine Feynman-technique explanation — the plain-language meaning woven
into the sentence where a term does its work, not segregated into a skippable
aside.

1. **Every technical term or code identifier gets its plain-English meaning
   in the same sentence it first appears in.** Not "`gatePlanning` and
   `gateRetro` take the project row and return." Instead: name what the two
   functions were supposed to do, in plain terms, before or immediately
   after naming them.
2. **An analogy or plain restatement precedes precise mechanism wherever the
   mechanism is non-trivial.** If a paragraph would lose a reader with no
   coding background, that reader needs a foothold before the technical
   description, not after.
3. **A code snippet gets a one-sentence plain-English "what this does"
   immediately before or after it.** Never let a code block stand as the
   only explanation of an idea.
4. **No bare complexity or math notation without a grounding example.** Not
   "O(n) per query." Instead, a concrete number: what that actually means at
   a size the reader can picture.
5. **A concept used more than once in a page is glossed once, not
   reflexively every time** — the fix for rule 1 is not to over-explain, it's
   to explain once, at first use, clearly enough that the reader doesn't need
   it again.
6. **The test**: read the paragraph as someone who has never written code.
   Is there a point where that reader would have to stop and guess what a
   term means to keep following the argument? If yes, the rule above it was
   skipped, not the reader's fault.

**Scope of rule 1, clarified after the first calibration pass (2026-09-02):**
"First appears" means first appears **on that page**, not first appears
anywhere on the site. The site explicitly offers non-linear reading paths
(`intro.md`'s "Show me a real system first," "Something is already wrong,"
"Thirty minutes") that send a reader into a chapter out of book order — a
term glossed only in an earlier chapter is unglossed for a reader who
started somewhere else. Each page must stand alone for its own key jargon.
A "Reference material" or "Where it shows up" link-list entry is exempt —
that text is a pointer to a page that will do the explaining, not a claim
being made in the current argument, so it can use a term at book-level
density (e.g. "verbalized confidence," "critic-revise loops").

## Applying both together

Voice and accessibility can pull in different directions — accessibility asks
for more explanation, voice rule 6 (no invented precision) and rule 4
(no restating captions) ask for less padding. When they conflict, accessibility
wins for a term a reader needs to keep following the argument; voice wins for
decoration. A plain-language gloss earning its place because a reader needs it
is not the same kind of addition as a sentence restating what a diagram
already showed.
