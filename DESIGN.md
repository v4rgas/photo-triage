# photo-triage — Design

Visual and interaction specification for stage 5, the browser UI. Companion to
[PROJECT.md](PROJECT.md), which owns the architecture, endpoints and data model.
Where the two disagree, PROJECT.md wins on behaviour and this document wins on
appearance.

---

## 0. The constraint that shapes everything

**No CDN. No bundler. No framework. No network at runtime.** Every font, every
byte of CSS and JS ships in `photo_triage/static/`. The tool must work on a
laptop in aeroplane mode, because the entire premise is that nothing leaves the
machine.

This is a gift, not a limitation. It rules out the entire category of design
that is assembled from imported dependencies, and forces choices that are
actually *ours*.

---

## 1. What this interface is for

A person sits down with 24,000 images and a decision to make about each one.
They will be here for forty minutes. They are going to hold `S` and `D` down
with their left hand and sweep the mouse with their right.

Three consequences, in priority order:

1. **The image is the interface.** Every pixel spent on chrome is a pixel not
   spent on the thing being judged. Chrome recedes; photographs dominate.
2. **Latency is the primary aesthetic.** A beautiful UI that stutters while
   scrolling a grid feels broken. A plain one that never drops a frame feels
   expensive. Perceived quality here is 70% frame timing.
3. **It must survive a long session.** High-contrast, high-saturation chrome is
   fatiguing after ten minutes and actively distorts colour judgement of the
   photos. The UI is deliberately desaturated so the images are the only
   saturated thing on screen.

The one thing someone should remember: *the toolbar is a sheet of frosted glass
that the photographs slide underneath.*

---

## 2. Aesthetic direction — "Liquid Glass"

Borrowed from the Apple design tradition, restated in our own terms. Four
principles, and each has teeth — they resolve real decisions later in this
document.

### 2.1 Deference

Chrome defers to content. The toolbar is translucent, the background is a
near-neutral that disappears behind imagery, and there is no ornament that does
not communicate state. If a decorative element cannot answer "what does the user
learn from this?", it is deleted.

### 2.2 Materiality

Surfaces behave like physical materials with consistent properties. Glass is
**translucent, blurred, and lit from above**; it is always *in front of*
content, never behind it, and it always casts a soft shadow onto what it
overlaps. Glass never appears flat against the page background — that reads as a
grey box, which is the failure mode of every glassmorphism knockoff.

There is exactly **one** glass material in this app, with one blur radius and
one tint, used in three places (§4.1). Multiple glass weights would read as
arbitrary.

### 2.3 Depth as hierarchy

Elevation communicates *permanence*, not importance:

| Layer | z | Material | Contents |
|---|---|---|---|
| 0 | — | opaque | the grid |
| 1 | 10 | glass | top toolbar (always present) |
| 2 | 20 | glass | selection bar (present only when a selection exists) |
| 3 | 30 | scrim + glass | lightbox |

Nothing else floats. No floating action buttons, no toast stack, no cards.

### 2.4 Motion with physics

Things that appear have come from somewhere. Motion is short, decelerating, and
never bouncy — real objects with mass do not overshoot and wobble. Full
treatment in §7.

---

## 3. Foundations

### 3.1 Colour

OKLCH throughout, for perceptually even lightness steps. Neutrals are tinted a
few degrees toward the accent hue so the greys feel related rather than dead.

**Dark only.** PROJECT.md §7 specifies a dark UI and that is the right call for
a reason worth recording: you are making judgements about photographs, and a
light surround washes out midtones and biases perceived exposure. Every gallery
and every editing application is dark for this reason. There is no theme toggle
and no light variant to maintain.

```css
:root {
  color-scheme: dark;

  /* Accent — a muted glacial blue. Used for selection and focus ONLY.
     It is the only chromatic colour in the chrome. */
  --accent:        oklch(66% 0.13 247);
  --accent-quiet:  oklch(66% 0.13 247 / 0.16);

  /* Semantic. Used sparingly — never as a fill for anything larger than
     a badge. Tuned for legibility on dark, not lifted from a light palette. */
  --junk:          oklch(68% 0.15 42);    /* warm — quarantine, JUNK */
  --review:        oklch(76% 0.13 85);    /* amber — REVIEW */
  --keep:          oklch(70% 0.12 152);   /* green — KEEP, protected */

  /* Neutrals, hue-matched to the accent. Never pure #000 — a true black
     surround makes dark photographs look like holes in the page. */
  --bg:            oklch(17% 0.008 247);
  --bg-sunken:     oklch(13% 0.008 247);  /* tile placeholder, review mode */
  --ink:           oklch(95% 0.004 247);
  --ink-quiet:     oklch(66% 0.008 247);
  --hairline:      oklch(95% 0.004 247 / 0.13);
}
```

Note what is absent: no gradients in the chrome, no glow, no neon, no purple.
The only gradient anywhere is the near-invisible lighting gradient on the glass
itself (§4.1), which exists to model a light source, not to decorate.

### 3.2 Typography

System-first, with one bundled variable font so the app looks identical on every
machine and still works offline.

```css
@font-face {
  font-family: 'Geist';
  src: url('./geist-variable.woff2') format('woff2-variations');
  font-weight: 100 900;
  font-display: swap;   /* local file — resolves instantly, swap never flashes */
}

--font-ui: 'Geist', system-ui, -apple-system, sans-serif;
```

Geist is SIL Open Font Licence, ships as a single ~40 KB variable woff2, and has
the tight apertures and tabular figures this UI needs. Bundle it; do not `@import`
it. (Apple's own SF Pro is licensed for Apple platforms only — we get the family
resemblance from `-apple-system` in the fallback stack on macOS and from Geist
everywhere else.)

Scale — small, tight, and deliberately limited to four sizes. This is a tool,
not a landing page; there is no display type anywhere.

| Token | Size | Weight | Tracking | Use |
|---|---|---|---|---|
| `--t-micro` | 11px | 560 | +0.02em | badges, counts on tiles |
| `--t-body`  | 13px | 440 | 0 | everything |
| `--t-input` | 15px | 440 | −0.005em | the search field |
| `--t-title` | 17px | 620 | −0.015em | lightbox filename, empty states |

Negative tracking on the larger sizes and positive on the smallest — optical
correction, the thing that separates typography that was set from typography
that was defaulted.

**Numbers use `font-variant-numeric: tabular-nums`.** Result counts update on
every keystroke; proportional figures make them jitter.

### 3.3 Space and radius

A 4px base, but *not* applied uniformly — rhythm comes from contrast between
tight and generous.

```css
--s1: 4px;  --s2: 8px;  --s3: 12px;  --s4: 20px;  --s5: 32px;
```

Tight (`--s1`/`--s2`) inside a control group; generous (`--s4`/`--s5`) between
regions. The grid gutter is `--s1` — 4px — because dense contact sheets read
better than airy ones, and because a tighter gutter fits more decisions on
screen.

Radius follows the nesting rule: an inner radius equals the outer radius minus
the padding between them, so corners stay concentric.

```css
--r-glass: 16px;   /* floating panels */
--r-control: 9px;  /* buttons, inputs inside a panel: 16 - 7 */
--r-tile: 6px;
```

---

## 4. Materials

### 4.1 Glass

One recipe. Used in exactly three places: the toolbar, the selection bar, the
lightbox chrome.

```css
.glass {
  background: oklch(21% 0.008 247 / 0.58);
  backdrop-filter: blur(28px) saturate(180%);
  border: 1px solid var(--hairline);
  border-radius: var(--r-glass);
  box-shadow:
    0 0 0 0.5px oklch(0% 0 0 / 0.4),
    0 12px 32px -8px oklch(0% 0 0 / 0.5);
  /* Isolate the blur so the browser doesn't re-rasterise the whole page. */
  contain: paint;
}

/* The specular edge: a 1px highlight along the top only, modelling a light
   source above. This single line is what makes it read as glass rather than
   as a translucent grey rectangle. */
.glass::before {
  content: '';
  position: absolute; inset: 0;
  border-radius: inherit;
  padding-top: 1px;
  background: linear-gradient(oklch(100% 0 0 / 0.16), transparent 1px);
  pointer-events: none;
}
```

The `saturate(180%)` is the part people leave out. Blur alone desaturates what's
behind it and the panel goes muddy over a colourful grid; boosting saturation
back up is what makes photographs *glow* through the glass instead of smearing.

**The performance rule, and it is not negotiable:**

> `backdrop-filter` appears **only** on fixed-position elements with a bounded
> area. It never appears on a tile, a badge, a hover state, or anything inside
> the scroll container.

A blurred element must re-composite whenever anything behind it changes. Behind
the toolbar, something is *always* changing — that's the grid scrolling past.
One fixed 64px-tall strip is a cost the compositor absorbs happily. Two hundred
blurred tiles is a slideshow. If a future contributor wants frosted tiles, the
answer is no, and this paragraph is why.

### 4.2 The grid

Opaque, flat, no card treatment. A tile is a photograph with a 6px radius — no
shadow, no border, no padding, no card wrapper, no caption bar beneath it.

```css
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(clamp(96px, 12vw, 168px), 1fr));
  gap: var(--s1);
  padding: var(--s2);
  padding-top: calc(64px + var(--s4));   /* clear the fixed toolbar */
}

.tile {
  aspect-ratio: 1;
  border-radius: var(--r-tile);
  background: var(--bg-sunken);          /* the placeholder before load */
  object-fit: cover;
  cursor: pointer;
  /* transform/opacity only — never layout properties */
  transition: transform 180ms var(--ease-out), opacity 180ms linear;
}
```

Density is user-controlled by a three-position control (S / M / L) that swaps
the `minmax` floor between 96, 132 and 168px. At S you fit roughly 200 tiles on
a 1440p screen, which is the density at which meme-sweeping actually works.

**Thumbnails are `loading="lazy"` and `decoding="async"`.** With 24,000 rows the
grid is windowed — render a viewport's worth plus two screens of overscan, and
recycle nodes on scroll. This is the single most important implementation detail
in the whole UI; get it wrong and nothing else here matters.

### 4.3 Tile metadata — tiered, not stacked

PROJECT.md §7 requires five pieces of information per tile: category badge,
similarity score, confidence, dimensions and source folder. Printed
simultaneously on a 96px tile, that is five lines of 11px text over a photograph
— it obscures the only thing the user is actually judging, and at 200 tiles on
screen it is 1,000 strings of text competing with 200 images.

So it is tiered by *when the user needs it*, not crammed into one state.

**Always visible — one glyph, bottom-left.** A 6px filled dot in `--junk`,
`--review` or `--keep`, over a short bottom-edge vignette so it survives on a
white photograph. Not a pill, not a word. At sweeping density you are reading
colour, not text, and a dot costs almost no pixels of photograph. The legend
lives in the filter dropdown where the same three colours appear.

**On hover — the numbers.** A bottom-anchored strip fades in over 90ms carrying
score (when searching) and confidence, both `--t-micro`, `tabular-nums`, white
on a `linear-gradient(transparent, oklch(0% 0 0 / 0.72))` vignette. It is a
gradient, not a blurred panel — see §4.1. Hovering is already how `S` and `F`
target a tile, so the information appears exactly when the cursor is committed
to that image.

**On demand — the rest.** Dimensions and source folder appear in the lightbox
and in review mode, where there is room to set them properly. They are context
for a decision already being made, not scanning signals.

At size **L**, the hover strip is always-on rather than hover-triggered: at 168px
there is room, and L is the size people use when they are reading rather than
sweeping. This is the only density-conditional behaviour in the UI.

> If a future contributor wants the full metadata always visible on every tile,
> the honest answer is that the requirement is really "I want to see this while
> judging" — and that is review mode (§4.5), which is built for it.

### 4.4 Selection

Selection is the state the user manipulates most, so it gets the strongest
visual treatment in the app — and it still uses only two properties.

```css
.tile[aria-selected='true'] {
  outline: 3px solid var(--accent);
  outline-offset: -3px;   /* inset, so selection never shifts layout */
  transform: scale(0.94);
}
```

The inward scale is the whole trick: selected tiles *recede slightly*, opening a
visible gutter around each one. A grid of 40 selected images reads instantly as
a set, from across the room, without a single checkbox. Unselected neighbours
never move, because `outline` and `transform` are both compositor-only.

Protected images (`S`) replace the category dot with a 13px shield glyph in
`--keep` — protection outranks category, and a saved image is no longer being
judged. Quarantined images in the quarantine view sit at `opacity: 0.55` with a
`--junk` hairline.

### 4.5 Review mode

PROJECT.md §7 makes this first-class and says not to skip it, so it gets a real
design rather than a bigger grid. It is a **different activity**: the grid is
for acting, review mode is for judging, and the two want opposite layouts.

Entered from any category chip in the filter row, or from the stats bar. Its job
is to make a model false-positive impossible to miss before a bulk delete.

| | Grid | Review mode |
|---|---|---|
| Question | "which of these do I act on?" | "is this category right?" |
| Tiles per screen | ~200 | ~24 |
| Metadata | one dot | filename, confidence, dimensions, folder |
| Order | search rank | **confidence descending** |
| Chrome | glass toolbar | opaque header, no glass |

Concretely:

- **Opaque `--bg-sunken` background, no glass anywhere.** Review mode is a
  destination, not an overlay. Dropping the translucency is what signals "you
  have changed activity" — and it removes the blur cost while large images are
  on screen.
- **Large tiles**, `minmax(240px, 1fr)`, `object-fit: contain` on
  `--bg-sunken` rather than `cover`. Cropping to a square is exactly how you
  miss that the "screenshot" is a photograph of a whiteboard.
- **Sorted most-confident first**, per the spec. The interesting region is the
  *tail*, so the confidence sequence is exposed directly: a 2px bar under each
  tile, width proportional to confidence, in the category colour. Scrolling to
  where the bars visibly shorten takes the user to the decision boundary without
  them having to read a single number.
- **A rule drawn across the grid** where confidence crosses the bulk-action
  threshold, labelled *"below here, the model is guessing."* Everything above it
  is what a bulk delete would take.
- Full metadata is set properly beneath each tile at `--t-micro`, `--ink-quiet`,
  because here there is room and it is the point.
- Selection, `S`, `F` and `D` all work identically. Review mode is not read-only
  — the whole value is fixing a mistake the moment you spot it.

The spec suggests requiring a category be reviewed once before it can be bulk
deleted. Support it in the UI honestly: until reviewed, the category's bulk
action is a ghost button labelled *"Review 4,182 first"* that opens review mode.
Not disabled-with-a-tooltip — a disabled control that won't say why is the most
frustrating pattern in interface design.

---

## 5. Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  ╭────────────────────────────────────────────────────────────╮  │ ← glass,
│  │ ⌕ birthday cake       [group▾] [cat▾] [folder▾] [show▾]    │  │   fixed,
│  │ 1,022 match · 7,244 on disk · 96 saved            S · M · L│  │   inset
│  ╰────────────────────────────────────────────────────────────╯  │
│                                                                  │
│   ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣   ← grid scrolls under it      │
│   ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣                                │
│   ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣                                │
│                                                                  │
│      ╭────────────────────────────────────────────────╮          │ ← glass,
│      │ 40 selected  ·  all 1,022   Save  Quarantine  ✕│          │   slides up
│      ╰────────────────────────────────────────────────╯          │   on select
└──────────────────────────────────────────────────────────────────┘
```

The spec's status bar is **merged into the toolbar as a second line** rather
than given its own band. Three stacked full-width regions before you reach a
single photograph is a lot of chrome for a tool whose entire premise is looking
at images, and the counts are glanceable context — they belong next to the
filters that produce them, not in a bar of their own. The action buttons the
spec puts in that bar move to the selection bar, where they are contextual: a
"Save" button means nothing with nothing selected.

The toolbar is a **floating inset panel**, not a full-bleed bar — 12px of grid
visible down both sides and along the top. This is the detail that sells the
material: you can see the photographs continue underneath and around it, so it
reads as a physical object resting on top rather than as a page header.

Layout is asymmetric and left-weighted. Search sits left and takes the space it
needs; filters cluster right; counts sit quiet on the second line in
`--ink-quiet` with `tabular-nums`. Nothing is centred except the selection bar,
which is centred because it is transient and needs to be found instantly.

### The two select-alls

`A` selects what is *shown*; the spec also needs "select all matching", which
reaches beyond the viewport via `/api/ids`. These are dangerously easy to
confuse — one acts on 300 images, the other on 1,022 — so they are never
rendered as two adjacent buttons.

Only `A` has a control. Selecting everything shown then reveals a single inline
link in the selection bar — *"40 selected · **select all 1,022 matching**"* —
which is the Gmail pattern, and it works because the escalation is offered only
once you have demonstrated intent. When the extended selection is active the
selection bar's count switches to `--junk` and reads *"all 1,022 matching"*, so
the wider blast radius is never ambiguous at the moment `D` is pressed.

### Responsive

Below 640px the toolbar collapses to search + a filter button that expands the
filter row inline beneath it (not a modal — see §8). The grid floor drops to
72px. Nothing is removed; the keyboard shortcuts are simply unreachable, and the
selection bar's buttons grow to 44px touch targets.

Use `@container` on the toolbar rather than `@media`, so it responds to its own
width and stays correct if the app is ever embedded.

---

## 6. Controls

Three tiers, and most controls are the quietest tier. Making everything a
primary button is how a tool ends up looking like a marketing page.

- **Ghost** (default) — transparent, `--ink-quiet` label, background fades to
  `--accent-quiet` on hover. Filters, density, view switches.
- **Filled** — `--accent` fill, white label. Exactly one on screen at a time:
  the confirming action in the selection bar.
- **Destructive** — `--junk` label on ghost, becoming a `--junk` fill only while
  a confirmation is pending. Quarantine and purge.

Focus is a 2px `--accent` ring at 2px offset, via `:focus-visible` only, so
mouse users never see it and keyboard users always do. It is never removed.

The search field is the visual anchor of the toolbar: `--t-input`, a leading
magnifier glyph in `--ink-quiet`, no border, background `--accent-quiet` at 40%
opacity, expanding its `flex-grow` on focus. Debounce 120ms — enough to skip
intermediate keystrokes, short enough to feel live against a cached matmul.

---

## 7. Motion

```css
--ease-out:   cubic-bezier(0.22, 1, 0.36, 1);    /* ease-out-quint */
--ease-inout: cubic-bezier(0.65, 0, 0.35, 1);

--d-instant: 90ms;    /* selection, hover — must feel like zero latency */
--d-quick:   180ms;   /* panels, badges */
--d-settle:  320ms;   /* lightbox */
```

Exponential ease-out only. No bounce, no elastic, no spring overshoot — a sheet
of glass that wobbles when it lands is not glass.

**Where motion is spent.** Not scattered micro-interactions; a few orchestrated
moments:

1. **Results arriving.** New tiles fade from 0 to 1 and rise 6px, staggered by
   row with a **12ms** delay per row, capped at 8 rows (~96ms total). Staggering
   individual tiles would take a second to finish and feel sluggish; staggering
   by row reads as a single sheet unfurling.
2. **The selection bar.** Enters `translateY(12px) → 0` with opacity, `--d-quick`.
   Exits at 60% duration — exits should always be faster than entrances, because
   the user has already decided.
3. **Quarantine.** Selected tiles scale to 0.86 and fade out over `--d-quick`,
   then the grid reflows. The tiles must visibly *leave*; a batch that just
   vanishes leaves the user unsure it worked.
4. **Undo.** Runs the same animation in reverse, from the position the tiles
   will occupy. This is the most reassuring 200ms in the app — it is what makes
   `D` feel safe enough to press quickly.
5. **Lightbox.** Scales from the tile's actual bounding rect to centre over
   `--d-settle`. Read the rect with `getBoundingClientRect()` and animate with
   `transform` — never `width`/`height`/`top`/`left`.

**Reduced motion.** Under `prefers-reduced-motion: reduce`, all transforms and
staggers are removed; opacity crossfades at `--d-instant` remain, because
state changes still need to be legible. This is not optional.

Hover on a tile: `scale(1.03)` at `--d-instant`, and nothing else. No lift, no
shadow bloom, no overlay slide-up.

---

## 8. Interaction principles

**Optimistic, always.** Selection, save and quarantine update the DOM
immediately and reconcile with the server response. A triage session is
thousands of decisions; a 40ms round-trip on each one is 90 seconds of dead time
and it destroys the rhythm. On failure, revert the tile and surface the error in
the selection bar — never a toast that steals focus.

**No modals.** The spec calls for a confirm dialog on `D`; implement it as the
selection bar transforming in place — "Quarantine 40 images?" with confirm and
cancel — not as a centred overlay. The user's eyes are already on the selection
bar; a modal moves their attention and blocks the grid they're deciding about.
The single exception is `purge`, which is genuinely destructive, irreversible,
and *should* be hard.

**The keyboard is the primary input.** Every shortcut in PROJECT.md §7 works
without focus management tricks. `?` opens a shortcut sheet — the one piece of
progressive disclosure, hidden until asked for.

**Empty states teach.** Never "No results."

- *Before first search:* "23,584 images ready. Try `screenshot`, `receipt`, or
  `people at a table` — it searches what the picture looks like, not the
  filename."
- *No matches:* "Nothing matches `xyzzy`. Search is ranked, not filtered — a
  broader phrase will always return something."
- *All triaged:* "Nothing left in this folder. 2,140 saved, 8,904 quarantined."

**Progress is a hairline.** During the embed stage, a 2px `--accent` bar along
the bottom edge of the toolbar, with an ETA in `--t-micro`. No spinner, no
percentage ring.

---

## 9. Accessibility

Not a section that gets cut. This is a tool for looking at things, which makes
it exactly the tool where visual assumptions bite hardest.

- Every tile is a `<button>` inside a `role="grid"`, with `aria-selected` and an
  `aria-label` of the filename plus category. Arrow keys move a roving tabindex.
- Selection is never conveyed by colour alone — the inset scale carries it too.
- **Category is never conveyed by the dot alone.** The dot is colour-only by
  design (§4.3), so the tile's `aria-label` and the hover strip both name the
  category in words, and review mode sets it as text. `--junk` (warm) versus
  `--keep` (green) is precisely the red-green axis, and their lightnesses are
  near-identical (68% vs 70%), so to a deuteranopic user the two dots are the
  same dot. Shape is what has to carry it: `keep` is a shield glyph, not a dot.
- Text contrast ≥ 4.5:1 **against the blurred backdrop at its worst case**, not
  against the panel tint in isolation. Test the toolbar over a white photograph
  and over a black one; the glass tint opacity floor of 0.58 exists for
  precisely this reason and must not be lowered for aesthetics.
- `prefers-reduced-transparency: reduce` → glass becomes an opaque `--bg` panel
  with the hairline retained. The layout must not shift when it does.
- Full keyboard operation with a visible focus ring, including the lightbox,
  which traps focus and restores it to the originating tile on `Esc`.

---

## 10. What this design is not

Recorded so the intent survives contact with future contributors:

- **Not glassmorphism.** Glass is three fixed panels, never tiles, never
  decoration. If you find yourself adding a fourth blur, you are decorating.
- **Not dark-mode-with-glowing-accents.** No glow, no neon, no gradient text, no
  cyan, no purple. One muted accent hue, used for state only.
- **Not card-based.** No card wraps a photograph. Photographs are the cards.
- **Not animated everywhere.** Five orchestrated moments (§7). Everything else
  changes instantly, because instant is the correct duration for feedback.
- **Not responsive by amputation.** The mobile layout adapts; it never hides
  functionality.
- **Not metadata-over-photographs.** One dot at sweeping density; everything
  else is tiered to hover, the lightbox, or review mode (§4.3). The grid's job
  is to show pictures.

---

## 11. Implementation checklist

- [ ] `static/style.css` — tokens, glass recipe, grid, controls. One file, no
      preprocessor, plain nested CSS (baseline in all current browsers).
- [ ] `static/geist-variable.woff2` — bundled, subset to Latin, `preload`ed.
- [ ] Windowed grid renderer: viewport + 2 screens overscan, recycled nodes.
- [ ] `IntersectionObserver` for thumbnail fetch, not scroll listeners.
- [ ] Selection as a `Set` of row ids; DOM reads batched, writes in one rAF.
- [ ] Verify: 60fps scrolling with 24k rows, measured in DevTools with the
      toolbar visible. **This is a release gate, not a nice-to-have.**
- [ ] Review mode as a separate view, opaque, `object-fit: contain`, sorted by
      confidence descending, with the threshold rule drawn (§4.5).
- [ ] Verify: toolbar text legible over a pure-white and a pure-black image.
- [ ] Verify: category is recoverable without colour — label, hover strip and
      review mode all name it in words.
- [ ] Verify: "select all matching" and `A` cannot be confused at the moment
      `D` is pressed — the selection bar states which is active.
- [ ] Verify: full triage flow keyboard-only, and again with reduced motion,
      reduced transparency, and 200% browser zoom.
