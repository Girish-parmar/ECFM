---
description: Generate the complete study package (notes, homework, solutions, PDF) for the next MFAAT lecture in sequence, or a specific one if named.
argument-hint: "[optional: Part N Session M — otherwise continues from .claude/lecture-progress.json]"
---

# Generate the next lecture package

You are generating a complete, classroom-quality lecture package for **one session** of the MFAAT course
(Master in Financial Analysis and Algorithmic Trading) documented in this repo's `docs/lessons/PART_*.md` files.
This command is self-contained — follow it even if you have no memory of earlier lectures in this series.

## Step 0 — Which lecture?

- If `$ARGUMENTS` names a Part and Session (e.g. "Part 2 Session 1", "P2S1", "2 1"), generate that one.
- Otherwise, read `.claude/lecture-progress.json` in the repo root (`{"last_part": N, "last_session": M}`).
  If it doesn't exist, the next lecture is **Part 1, Session 1**.
  Otherwise: find how many sessions Part `N` has by counting `#### S<n> ·` headings under `## 4. Session-by-Session
  Plan` in `docs/lessons/PART_0N_*.md` (glob it; the numeric prefix is always 2 digits, `docs/lessons/PART_*.md`
  sorted gives you Parts 1–12 in order). If `M` was the last session of Part `N`, the next lecture is
  **Part N+1, Session 1**; otherwise it's **Part N, Session M+1**. If Part N was 12 and M was its last session,
  the course is complete — say so and stop; do not fabricate a Part 13.

## Step 1 — Read the real source material

Open `docs/lessons/PART_0N_*.md` and read, at minimum:
- The session's own block under `#### S<M> · <Title>` (Theory / Live coding / Worked examples / Lab / Homework —
  field names vary a little by part, use whatever's there).
- `## 2. Prerequisites & Setup` (or the part's equivalent) and the weekly overview, for context.
- Anything the session block references (a worked example fully spelled out elsewhere in the doc, a clinic it
  feeds into, an artifact-map row).

**Everything you teach must trace back to this real content.** Where the real session is light on quantitative
formulas (common in Part 1's early orientation sessions), don't invent formulas to fill space — teach the
frameworks/classifications that are actually there, and say so.

## Step 2 — Build the four files, following this exact structure

Naming: `Part<NN>_S<MM>_01_Lecture_Study_Notes.md`, `..._02_Homework_Assignment.md`, `..._03_Homework_Solutions.md`,
`..._04_Complete_Lecture_Package.pdf` (zero-padded 2-digit Part and Session numbers). Output directory: your
scratchpad, under `lectures/Part<NN>_S<MM>/` — never the repo working tree (these are the learner's personal study
files, not repo content).

**File 1 — Lecture Study Notes** (Markdown), in this order:
0. Prerequisite table (No./Prerequisite/Why Required/Required Knowledge/Revision Notes) + a 5-question
   self-check with answers, tailored to *this* session (build on the previous session's self-check where relevant,
   don't just repeat generic finance basics every time).
1. **A. Lecture Overview** — title, module/part/session/week, format, tools, learning objectives (numbered, tied
   to what's actually taught), expected outcomes, a topics/subtopics table with a time budget summing sensibly
   toward the session's real duration (90 min for Part 1; check the part's own format line for other parts).
2. **B. Concept-by-Concept Teaching** — one `B.n` subsection per major concept in the session (usually 3–6). For
   **each** concept give, as its own labelled parts: (1) definition, (2) why it exists / what problem it solves,
   (3) how it works step by step, (4) terminology, (5) formulas/principles/rules — real ones from the source
   material or the field, never invented, (6) practical applications & limitations, (7) common misconceptions.
3. **C. Examples and Worked Problems** — an everyday analogy, a real-world example, a numerical/technical example,
   a step-by-step worked problem, and one advanced case study, for the session's concepts as a set (not
   necessarily one of each per concept). **Any invented number must be explicitly labelled "illustrative/
   hypothetical, not real data."** A case study using a real historical event must be a genuinely well-documented
   one — hedge appropriately ("roughly," "according to X") and never invent precise statistics you're not
   confident of, especially anything that could be recent enough to fall after your knowledge cutoff.
4. **D. Lecture Notes and Revision Material** — key one-line definitions, the session's real formula(s) (skip
   this subsection's formula box entirely if the session genuinely has none — don't manufacture one), a
   comparison table, one clarifying diagram or table, a common-mistakes table (session-specific, not the part's
   whole mistake list), a quick revision summary, a one-page cheat sheet.
5. **Classroom Teaching Plan** — a time-boxed table (Time/Activity/Content/Example/Student Outcome) that sums to
   the session's real duration, plus 2–4 interactive check-in questions and a "connecting forward" paragraph
   linking to the next sessions/parts that build on this one (check the lesson plan's own forward references).

**File 2 — Homework Assignment** (Markdown, no answers): Section A Basic Understanding (5 questions, easy,
definitions/recall), Section B Conceptual Understanding (5 questions, medium, explain/compare/reason), Section C
Practical Problems (5 questions, medium–hard, calculation/application — reuse the session's real lab/homework task
where the source material already has one, don't replace it with something invented), Section D Advanced Thinking
(2 questions, hard: one case-study analysis, one open-ended), Section E Mini Project (one deliverable that extends
the previous session's, where the course structure implies continuity — e.g. a running research log). Give every
question a difficulty tag, the objective it tests, and marks (A=5×5=25, B=5×5=25, C=5×6=30, D=2×10=20, total 100;
Section E graded separately). If the session's own real homework/lab needs the learner's *own current* data
(a live account, today's real economic calendar, a live price) that you cannot know or fabricate, say so
explicitly in the question and design it to be completed with the learner's own live tools — never substitute
invented "current" data presented as real.

**File 3 — Homework Solutions**: full worked answer for every question (numbers computed and checked by you,
not asserted), the reasoning, the concept it reinforces, at least one common wrong approach, and a marking
rubric/rewritten checklist for open-ended or mini-project items. Every claimed number must actually be correct —
recompute arithmetic before writing it down.

**File 4 — Combined PDF**: build one HTML file from the three Markdown files with
`python3 tools/lecture-notes/md2pdf.py <out.html> "<cover title h1>" "<cover subtitle>" "<cover meta line>"
"Lecture Notes:<path to file 1>" "Homework:<path to file 2>" "Solutions:<path to file 3>"`, then render it with
headless Chromium: `/opt/pw-browsers/chromium --headless --disable-gpu --no-sandbox --no-pdf-header-footer
--print-to-pdf=<out.pdf> "file://<out.html>"`. Sanity-check it (a tall screenshot of the HTML, cropped and viewed
with the Read tool, is enough — no `pdftoppm` is installed) before delivering. Delete any preview PNGs and the
intermediate `combined.html` from the delivery folder afterward, but you can leave `combined.html` if useful for
your own debugging on a later invocation — just don't send it to the user.

## Step 3 — Deliver and update state

- `SendUserFile` all four files (not `combined.html`), status `proactive`, with a one-line caption naming the
  Part/Session/title.
- Write/update `.claude/lecture-progress.json` in the repo to `{"last_part": N, "last_session": M}` for the
  lecture you just generated. Commit it (message like `Advance lecture progress to Part N Session M`) and push
  to the current branch — this file is small and cheap to keep current so a future session (or a fresh container)
  knows where the series is without replaying this whole conversation. If a push fails, don't block delivery on
  it — just say so.
- In your reply, briefly summarize what the lecture covered, repeat any "labelled as illustrative" or
  "needs your own live data" caveats from Step 2, name the next lecture in sequence, and ask whether to continue
  — unless `$ARGUMENTS` said to keep going automatically, in which case proceed to the next one directly.
