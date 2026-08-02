# photo-triage — Coding Style

Extracted from John Ousterhout, *A Philosophy of Software Design* (2018), and
restated for this codebase (Python backend, dependency-free vanilla JS UI).

Companion to [PROJECT.md](PROJECT.md) (architecture, behaviour) and
[DESIGN.md](DESIGN.md) (appearance). Where they disagree with this document,
they win on *what* to build; this document wins on *how the code is shaped*.

---

## 0. The one idea

**Complexity is anything about the structure of a system that makes it hard to
understand or modify.** It is not measured by how it felt to write, but by how
it feels to the next person to read. It accumulates incrementally — no single
change makes a system complex, so no single change is exempt from care.

Complexity shows up as three symptoms. Learn to name them, because naming one
in a code review is more useful than "this feels messy":

| Symptom | Meaning |
|---|---|
| **Change amplification** | A simple conceptual change requires edits in many places. |
| **Cognitive load** | You must hold a lot in your head to make a correct change. |
| **Unknown unknowns** | You cannot tell which code you must change, or what you must know. The worst of the three — the others announce themselves; this one lets you ship a bug. |

Two causes: **dependencies** (code that cannot be understood in isolation) and
**obscurity** (important information that isn't apparent). Every rule below is
an attack on one of those two.

> Ousterhout: *"If a software system is hard to understand and modify, then it
> is complex. If it is easy to understand and modify, then it is simple."*

---

## 1. Strategic, not tactical

Tactical programming is optimising for "working code, right now". It always
wins the sprint and always loses the quarter, because each shortcut is small
and their sum is a system nobody wants to touch.

**The rule for this repo:** working code isn't enough. Spend roughly **10–20%
of the time on each change** making the *design* better than you found it — a
better interface, a comment that removes a surprise, a special case eliminated.
This is investment, not gold-plating, and it is paid continuously in small
amounts, never as a "cleanup sprint" (which never comes).

A concrete test before opening a PR: *is the codebase's design better after
this change than before it?* If the answer is "neutral, I just added a
feature", look again for the small investment you skipped.

This applies with full force to the prototype-derived code. PROJECT.md §9
Pitfalls exists because the prototype was written tactically. We are rebuilding
precisely to not do that again.

---

## 2. Modules should be deep

A module is any unit with an interface and an implementation: a Python module,
a class, a function, a JS file. Its **cost** is its interface (what callers must
learn); its **benefit** is the functionality it hides.

**Deep** = powerful implementation behind a simple interface. **Shallow** =
interface nearly as complicated as the implementation, so the abstraction pays
for nothing.

```python
# Deep: the caller learns one function. Behind it: mmap, dtype checks,
# L2-normalisation, row-count/paths.json consistency, corruption recovery.
def load_embeddings(cache_dir: Path) -> Embeddings: ...

# Shallow: the caller now knows about mmap modes, dtypes and normalisation,
# and must sequence three calls correctly. Nothing was hidden.
def open_embed_file(path, mode): ...
def parse_embed_header(fh, dtype, expected_dim): ...
def normalise_rows(arr, in_place): ...
```

Rules that follow:

- **Prefer fewer, larger modules over many small ones.** "Classitis" — the
  belief that classes and functions should be as small as possible — produces
  many shallow modules, and complexity comes from the *accumulation of
  interfaces*, not from the length of any one file. A 200-line function with one
  clear job and no repeated code is not a problem; five 40-line functions that
  must be called in the right order are.
- **A simple interface beats a simple implementation.** If a messy
  implementation buys a clean interface for many callers, take that trade. The
  interface is paid for by every caller, forever; the implementation is paid
  for once, by you.
- **Design the interface for the common case.** The 95% call should need no
  optional arguments, no setup object, no ordering knowledge. Rare needs get
  extra parameters or a separate entry point — never a mandatory step for
  everyone.

**Red flag — Shallow Module:** the interface for a class or function isn't much
simpler than its implementation.

---

## 3. Information hiding and leakage

The single most important technique for depth: each module **encapsulates a
design decision** that appears nowhere else.

**Information leakage** is when one design decision is reflected in multiple
modules — so changing it means changing all of them. It is the direct cause of
change amplification. Leakage is often invisible: it happens through shared
assumptions, not just shared code.

Concrete examples in this system:

- The `.jsonl` line format for `index.jsonl` is known **only** to the module
  that reads and writes it. No other stage does `json.loads(line)` on it.
- The float32 / N×512 / L2-normalised layout of `embeds.npy` is known only to
  the embedding cache module. The classifier asks for a matrix; it never learns
  the dtype or that normalisation already happened.
- The quarantine mirroring rule (how an original path maps to a quarantine
  path) lives in one function. If the UI, the mover and the undo path each
  compute it, that's leakage — and a future change to the rule silently breaks
  undo.

**Temporal decomposition** is the most common cause of leakage: structuring
modules by *the order things happen* rather than by *what knowledge they hold*.
Note that our five stages (scan → embed → classify → …) are a *pipeline of
executables*, not a licence to decompose each stage's internals by time. Within
a stage, structure by knowledge: a "read the file" class and a "write the file"
class both know the format — merge them.

> **Order of execution is rarely the right basis for decomposition.**

**Red flags:** Information Leakage · Temporal Decomposition · Overexposure (an
API forces callers to be aware of rarely used features in order to use common
ones).

---

## 4. General-purpose modules are deeper

Aim for **somewhat general-purpose**: the interface should be general enough to
cover several plausible uses, while the implementation solves today's problem.

The push toward specialisation is what produces shallow, leaky code — a
`hide_selected_memes_and_refresh_grid()` bakes today's UI flow into what should
be a selection primitive.

Four questions before finalising an API:

1. **What is the simplest interface that covers all my current needs?** If
   reducing the API's method count increases the work at call sites, that's the
   wrong direction.
2. **In how many situations will this method be used?** A method built for one
   call site is probably too special-purpose.
3. **Is this API easy to use for my current needs?** If not, it's too general.
4. **Can I describe it without referring to the caller?** If the doc for a
   selection module mentions "meme mode", the caller has leaked into it.

**Separate general-purpose from special-purpose code.** Special-purpose code
belongs *above* the general-purpose layer that it uses, never mixed into it.

**Red flag — Special-General Mixture.**

---

## 5. Different layer, different abstraction

Adjacent layers should have **different** abstractions. When they don't, the
layer is not earning its place.

- **Pass-through methods** — a method that does almost nothing but forward its
  arguments to another method with a similar signature — add interface, hide
  nothing. Delete the layer, expose the lower method directly, or genuinely
  redistribute the work. (Exception: a decorator or an adapter that meaningfully
  *changes* the abstraction is fine.)
- **Pass-through variables** — a parameter threaded through five call frames so
  the sixth can use it — force every intermediate layer to know about something
  irrelevant to it. Fix with a shared context object, or by giving the
  intermediate layers access to an object that already holds it. In the HTTP
  layer, prefer one request-scoped context over threading `cache_dir` through
  every handler.

**Red flag — Pass-Through Method.**

---

## 6. Pull complexity downwards

When there's an unavoidable ugliness, it is better for one module implementer
to suffer it than for every caller to.

> It is more important for a module to have a simple interface than a simple
> implementation.

This is the rule that decides most configuration questions here. **A
configuration parameter is complexity pushed upward** — it makes the user (or
the caller) solve a problem you were better placed to solve, and most users will
guess worse than you can measure. Before adding a knob, ask: *can I compute a
good value automatically?*

- Embedding batch size: derive from available VRAM. Don't ask.
- Classification thresholds: ship measured defaults from the 23,584-image
  corpus. An override may exist for experimentation; it must not be required.
- Grid page size: derive from viewport. Don't ask.

**Taking it too far:** don't pull complexity down into a module that has no
business knowing about it. The test is whether the complexity is *closely
related* to the module's existing job, and whether many callers benefit. A
config knob that genuinely only the user can answer (which folder to scan) stays
at the top.

---

## 7. Better together or better apart?

Default to **together** when any of these hold, and apart otherwise:

- They **share information** — both know the `.jsonl` format, both know the
  quarantine path rule.
- Joining **simplifies the interface** — two calls that are always made in
  sequence become one that cannot be sequenced wrongly.
- It **eliminates repetition** — the same nontrivial fragment appearing in
  several places usually means a missing abstraction, not a missing copy.
- The pieces are **hard to understand separately** (Conjoined Methods).

Keep apart when the code is general-purpose vs special-purpose, or when the
two halves genuinely share nothing.

**On splitting functions:** split when a *subtask is cleanly separable* — a
child with a crisp interface that a reader can understand without reading the
parent. Do **not** split just to hit a line count; that produces conjoined
methods that must be read as a pair, which is worse than the long version.
Splitting into two functions each of which returns half the answer to the same
question is almost always wrong.

**Red flags:** Repetition · Conjoined Methods.

---

## 8. Define errors out of existence

Exceptions are the single largest source of complexity in most systems: the
error path is rarely tested, and every `except` in a caller is interface the
caller had to learn.

**The best exception is one that cannot occur** — change the semantics so the
condition is simply normal behaviour.

Techniques, in preference order:

1. **Define it away.** `unset` on a nonexistent key does nothing rather than
   raising. Deleting an image already gone from disk succeeds. A search with
   zero matches returns an empty list — it is not an error.
2. **Mask it.** Handle the condition inside the module so callers never learn
   it existed — a truncated `embeds.npy` triggers a rebuild of the missing rows
   rather than surfacing a corruption error to the UI.
3. **Aggregate it.** Handle many exceptions in one place high up, rather than
   at each site. An unreadable image during a 24,000-file scan is recorded and
   skipped by the scan loop; every per-file decoder does not need its own
   recovery policy. The HTTP layer gets **one** top-level handler that turns any
   unhandled exception into a 500 with a logged traceback.
4. **Just crash.** For genuinely unrecoverable conditions (out of memory, no
   CLIP weights on disk), print a clear message and exit. Elaborate recovery for
   conditions that cannot be recovered from is pure cost.

**Taking it too far:** don't define away errors that the caller genuinely needs
to know about. Silently ignoring a request the user asked for — a move that
didn't happen — is worse than raising. The test: does defining it away discard
information the caller needs?

Apply the same thinking to **special cases**. Every `if` for a special case is
complexity in every reader's path. Prefer designs where the general case
subsumes it: an empty selection follows the same code path as a selection of
one, and "no filter active" is the filter that matches everything.

---

## 9. Design it twice

For anything nontrivial — a module boundary, a cache format, an interaction
model — **sketch two or three genuinely different designs** before choosing.
Make them radically different, not variations; even a design you reject
clarifies what matters about the one you keep.

The comparison is on: *which interface is simpler for callers? which is more
general? which is more efficient?*

This is not a luxury. It is the cheapest possible time to discover that the
obvious design was the wrong one. Being smart is not a substitute for
considering alternatives — the first idea is rarely the best, for anyone.

---

## 10. Comments

Comments capture the design information that *cannot* be expressed in code —
the "why", the invariants, the units, the ranges. Without them, the abstraction
is incomplete, because readers must read the implementation to use the
interface, and then the abstraction has failed.

The four excuses are all wrong: good code is *not* fully self-documenting
(informal design reasoning does not fit in code); you do have time; comments
that go stale are a maintenance failure, not an argument against comments; and
worthless comments are an argument for writing better ones.

**Rules:**

- **Don't repeat the code.** If a comment carries no information not visible in
  the line beside it, delete it.
  ```python
  # BAD — repeats the code
  # Increment i.
  i += 1

  # GOOD — states what the code can't
  # Rows are appended in scan order; index.jsonl line N corresponds to
  # embeds.npy row N. Nothing downstream re-sorts, so this must stay true.
  ```
- **Use different words than the code.** If your comment reuses the identifiers
  from the line it describes, it probably repeats it. A comment should add
  precision (below the code) or intuition (above it).
- **Lower-level comments add precision** — units, boundary conditions,
  null/None meaning, whether a range is inclusive, who owns a resource.
  `similarity: float` says nothing; "cosine similarity in [-1, 1]; both vectors
  are L2-normalised, so this is a plain dot product" says everything.
- **Higher-level comments add intuition** — what the block accomplishes, and
  why, omitting the details.
- **Interface comments describe *what*, never *how*.** If a docstring describes
  the implementation, a caller has been given information they don't need and
  will come to depend on. **Red flag — Implementation Documentation
  Contaminates Interface.**
- **Implementation comments describe *what* and *why*, not *how*.** The code
  shows how.
- **Cross-module design decisions** get documented in one central place — a
  module docstring or a `docs/` note — with short pointers from the code that
  depends on it. Duplicating the explanation guarantees the copies diverge.

**Write the comments first.** Before implementing a class or a function, write
its interface comment. This is a design tool, not documentation overhead: if the
comment is long, awkward, or full of "and also", the abstraction is wrong and
you have found out before writing a line. Comments written afterwards are
written under pressure to be done, and they are worse.

**Red flags:** Comment Repeats Code · Implementation Documentation Contaminates
Interface · **Hard to Describe** (if the doc for a variable or method must be
long to be complete, the design is probably wrong).

---

## 11. Naming

Names are a form of abstraction — a good one creates an image in the reader's
mind and rules out what the thing is *not*.

- **Precise.** `count` is vague; `pending_embed_count` isn't. Avoid `data`,
  `info`, `result`, `value`, `obj`, `handle`, `process()`, `manage()`.
- **Consistent.** One concept, one name, everywhere, and that name is never
  reused for anything else. In this codebase: a `path` is always an absolute
  `Path`; a `rel` is always a string relative to the scan root; a `row` is
  always an integer index into `embeds.npy`. Never a "path" that is sometimes
  relative.
- **Loop variables:** in a nested loop, `i`/`j` invite exactly the class of bug
  Ousterhout opens the chapter with. Name them: `for row in …`, `for prompt_i in
  …`.
- **Length scales with scope.** A three-line comprehension can use `p`. A module
  global cannot.

**Red flags:** Vague Name · **Hard to Pick Name** — difficulty naming something
is evidence that the entity does more than one thing, or that its boundaries are
in the wrong place. Treat it as a design signal, not a vocabulary problem.

---

## 12. Consistency

Consistency lowers cognitive load: once you have learned how one thing works,
you know how the others work. Inconsistency is worse than a slightly inferior
convention applied uniformly.

For this repo:

- **Python:** `snake_case`, type hints on every public function, `pathlib.Path`
  for all filesystem work (never string concatenation), f-strings, `logging`
  never `print` (except deliberate CLI output).
- **JS:** ES modules, `const` by default, no build step, no framework, no CDN.
  Event handlers named `on<Thing><Event>`.
- **Cache files:** every stage writes atomically (temp file + `os.replace`) and
  is resumable. No stage may leave a half-written cache on interrupt.
- **The rule:** when modifying existing code, follow the existing convention
  even if you prefer another. Change conventions only with a deliberate,
  repo-wide migration — "I'll do it the new way from here" is how a codebase
  ends up with two of everything.

**Taking it too far:** consistency means treating *similar* things similarly. Do
not force genuinely different situations into one convention just to make them
match.

---

## 13. Code should be obvious

Obvious means: a reader understands the behaviour and meaning at a quick
reading, and their first guess about it is right.

Obviousness is a property of *readers*, not authors — you cannot judge your own
code's obviousness. Code reviews are the mechanism: if a reader says it isn't
obvious, it isn't, whatever the author thinks.

**Makes code obvious:** precise names · consistent conventions · judicious
whitespace and blank lines separating blocks · comments that supply what the
code can't.

**Makes code less obvious:**

- **Generic containers.** Returning a tuple or dict where the caller writes
  `result[0]` / `result["value"]` hides meaning. Define a `NamedTuple` or
  `@dataclass` with real field names and a docstring. Slightly more work to
  write; far less to read.
- **Declared type differs from allocated type** — declare what you actually
  built.
- **Code that violates reader expectations** — anything that starts a thread,
  keeps running after return, mutates its argument, or blocks. If you can't
  avoid it, document it at the surprise.
- **Event-driven code** — including our UI's handlers and any async work. The
  flow of control is invisible, so each handler's comment must say *when it is
  invoked and by whom*.

> **Software should be designed for ease of reading, not ease of writing.**

**Red flag — Nonobvious Code.**

---

## 14. On tests, TDD and patterns

- **Unit tests are essential**, specifically because they enable refactoring.
  Without them, nobody dares improve a design, and the system ossifies. Test
  coverage on the cache-format and path-mapping modules is non-negotiable —
  those are the ones a redesign will touch.
- **TDD is rejected as the primary loop.** It focuses on getting one feature
  working at a time, which is tactical by construction. **The increments of
  development should be abstractions, not features**: when you need a new
  feature, figure out the best design for it first, then write the code. (Fixing
  a bug is the exception — writing the failing test first is right there.)
- **Design patterns are a tool, not a goal.** Over-application is as harmful as
  the problem it solves. If a natural solution exists, use it.
- **Getters and setters are shallow methods.** Don't add them reflexively; in
  Python, expose the attribute and add a `property` only when there is real
  behaviour.
- **Inheritance:** prefer composition or interfaces. Implementation inheritance
  creates dependencies between parent and child that make both harder to change.

---

## 15. Performance

- **Simpler code is usually faster code** — fewer layers, fewer special cases,
  fewer instructions.
- **Measure before modifying, and measure after.** Intuition about hot spots is
  unreliable. Our known critical paths, from PROJECT.md, are the embedding pass
  (one-off, GPU-bound) and the similarity query (must feel instant), plus the
  UI's frame timing, which DESIGN.md names as the primary aesthetic.
- **Design around the critical path.** For the few places that genuinely matter,
  ask "what is the minimum work this must do?" and design for that directly,
  rather than bolting special cases onto a general path. Everywhere else,
  optimise for clarity and do not think about performance at all.

---

## 16. The checklist

Before opening a PR, scan for these. The presence of any one is a signal, not a
verdict — but it demands a justification.

- [ ] **Shallow Module** — interface nearly as complex as the implementation.
- [ ] **Information Leakage** — one design decision known to several modules.
- [ ] **Temporal Decomposition** — structure follows execution order, not knowledge.
- [ ] **Overexposure** — common usage forces awareness of rare features.
- [ ] **Pass-Through Method** — forwards args, adds nothing.
- [ ] **Repetition** — a nontrivial fragment appears again and again.
- [ ] **Special-General Mixture** — special-purpose code inside general-purpose code.
- [ ] **Conjoined Methods** — you can't understand one without the other.
- [ ] **Comment Repeats Code** — the comment says what the line already says.
- [ ] **Implementation Documentation Contaminates Interface** — docstring leaks internals.
- [ ] **Vague Name** — conveys little.
- [ ] **Hard to Pick Name** — the entity's boundaries are wrong.
- [ ] **Hard to Describe** — complete documentation would have to be long.
- [ ] **Nonobvious Code** — can't be understood in a quick read.

And the two questions that matter most:

1. *Is the design of this codebase better after my change than before it?*
2. *Would a reader who has never seen this file guess right?*

---

## Appendix — the 15 principles, verbatim

1. Complexity is incremental: you have to sweat the small stuff.
2. Working code isn't enough.
3. Make continual small investments to improve system design.
4. Modules should be deep.
5. Interfaces should be designed to make the most common usage as simple as possible.
6. It's more important for a module to have a simple interface than a simple implementation.
7. General-purpose modules are deeper.
8. Separate general-purpose and special-purpose code.
9. Different layers should have different abstractions.
10. Pull complexity downward.
11. Define errors (and special cases) out of existence.
12. Design it twice.
13. Comments should describe things that are not obvious from the code.
14. Software should be designed for ease of reading, not ease of writing.
15. The increments of software development should be abstractions, not features.
