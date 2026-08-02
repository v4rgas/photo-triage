# photo-triage

A local, offline tool for finding the photos that matter inside a folder full of
junk. Point it at a directory, it embeds every image with CLIP, and gives you a
browser UI where you can **search your photos with plain English**, select in
bulk, and move the rubbish into a quarantine folder that mirrors your original
structure.

Nothing leaves the machine. No API keys, no cloud, no upload.

> **Status:** specification for a clean-room rebuild, derived from a working
> prototype that triaged ~24,000 WhatsApp images. The prototype's mistakes are
> recorded in [Pitfalls](#9-pitfalls-read-this-before-writing-code) — that
> section is the most valuable part of this document.

**Scope:** finding and removing unwanted images. Explicitly *not* a photo
manager, deduplicating backup tool, or metadata repairer. It filters; you decide.

---

## 1. The problem

A phone backup folder contains 20 GB of images. Maybe 40–60% is memes,
screenshots, stickers, forwarded promotions and shop listings. The rest are real
photographs you would be upset to lose. There is no metadata separating them —
messaging apps strip EXIF on send, and filenames are just `IMG-20240512-WA0158.jpg`.

Sorting this by hand means looking at 24,000 images. Sorting it by rule means
writing rules that don't exist. The only signal is what the picture *looks like*.

## 2. The approach

Run a vision-language model (CLIP) locally over every image once, cache the
embeddings, then do everything else against that cache:

- **Zero-shot classification** — score each image against prompts like
  `"a screenshot of a phone screen"` to get a rough category.
- **Semantic search** — type `birthday cake` or `photos of my dog`, encode it
  with the same model, rank every image by cosine similarity. Instant, because
  the image side is precomputed.
- **Visual similarity** — "find more like this", which is how you clear a meme
  *and every re-send of it* in one sweep.

The embedding pass is the only expensive step and it is a one-off. Everything
after is a dot product against a matrix that fits in RAM.

### Why this beats the obvious alternatives

| Approach | Why it fails |
|---|---|
| EXIF metadata | Messaging apps strip it. In the prototype only **2.5%** of images had camera EXIF. |
| Filename patterns | Encode transmission date, not content. Say nothing about the picture. |
| File size / dimensions | Memes and real photos overlap heavily. |
| Perceptual-hash dedup alone | Found only **3.6%** redundancy. Useful, not the main win. |
| OCR text density | Genuinely good for memes, but ~50× slower than CLIP for a weaker signal. Add later as a *secondary* score. |

The measured payoff: on 23,584 WhatsApp images, CLIP flagged 44.5% as
memes/screenshots/stickers/text-graphics, and spot-checks confirmed those two
largest categories were reliable.

---

## 3. Architecture

Five stages. Each writes a cache file; each runs independently and resumably.

```
   ┌──────────────┐
   │ 1. scan      │  walk folder → dimensions, phash, dhash, colour count
   └──────┬───────┘  → index.jsonl
          │
   ┌──────▼───────┐
   │ 2. embed     │  CLIP ViT-B/32 image embeddings (GPU)
   └──────┬───────┘  → embeds.npy (float32, N×512, L2-normalised)
          │          → paths.json  (row id → relative path)
   ┌──────▼───────┐
   │ 3. classify  │  cosine vs averaged text-prompt vectors
   └──────┬───────┘  → scored.json
          │
   ┌──────▼───────┐
   │ 4. thumbs    │  256px JPEGs so the UI is instant
   └──────┬───────┘  → .phototriage/thumbs/<row>.jpg
          │
   ┌──────▼───────┐
   │ 5. serve     │  Flask + vanilla-JS single page
   └──────────────┘  → http://127.0.0.1:5000
```

**Critical invariant:** row id *i* is the *i*th line of `index.jsonl`, and the
same integer keys `paths.json`, `scored.json` and the thumbnails. Every other
structure keys off that integer row id. Never re-sort one without the other.

A row owns a **contiguous span of `embeds.npy`** rather than one row of it: a
photograph owns a single vector, a video owns one per sampled frame. The span
is derived by running down `index.jsonl` summing `segments`, so it cannot fall
out of step with the index, and for a folder of photographs every span has
length one and row *i* is still vector *i*. Video search takes the **best**
segment rather than their mean; see §11.5 and `video.py` for why averaging a
clip destroys what you were searching for.

### Measured timings (23,584 images, Ryzen 7 5700G + RX 6700 XT)

| Stage | Time | Notes |
|---|---|---|
| scan | 108 s | 14 processes, decode-bound |
| embed (GPU) | 78 s | ~300 img/s, **0.8 GB VRAM** |
| embed (CPU) | >20 min | aborted; ~20× slower |
| classify | <5 s | text encoding + one matmul |
| thumbs | 65 s | 186 MB output |

The GPU is barely working — batch 128 at 224×224 uses under a gigabyte. JPEG
decoding is the bottleneck, not inference. If you optimise anything, optimise
decode (`pillow-simd`, or `torchvision.io.decode_jpeg` on device).

---

## 4. Tech stack

Deliberately boring and dependency-light.

| Concern | Choice | Why |
|---|---|---|
| Model | `open_clip`, **ViT-B-32**, pretrained `laion2b_s34b_b79k` | Small, fast, strong zero-shot. 512-d. |
| Tensor runtime | PyTorch | CUDA / ROCm / MPS / CPU from one codebase — see [§4a](#4a-gpu-support) |
| Images | Pillow | set `Image.MAX_IMAGE_PIXELS = None` |
| Hashing | `imagehash` (phash + dhash) | near-duplicate detection |
| Server | Flask | one file, no build step |
| Frontend | Vanilla JS + CSS grid | **no framework, no bundler, no CDN** — must work offline |
| Packaging | `uv` | fast, reproducible |

## 4a. GPU support

The embed stage is the only GPU-bound work, and it's a one-off. The tool must
run acceptably on anything and get out of the way — a user with no GPU should
still be able to triage a folder, just slower.

### The central gotcha

**On AMD ROCm, PyTorch still uses the `torch.cuda` API.** `torch.cuda.is_available()`
returns `True`, `.cuda()` moves tensors to the Radeon, `torch.cuda.get_device_name()`
works. There is no `torch.rocm`. ROCm builds are a drop-in reimplementation of
the CUDA surface.

You therefore cannot tell the backends apart by the API. Use the build flags:

| | `torch.version.cuda` | `torch.version.hip` |
|---|---|---|
| NVIDIA build | `"12.4"` | `None` |
| ROCm build | `None` | `"6.3.42134-..."` |
| CPU-only build | `None` | `None` |

Write device-agnostic code against `torch.cuda`, and use `torch.version.hip`
**only** for logging and for applying the AMD-specific environment workaround.

### Detection

```python
import os, torch

def pick_device():
    """Returns (torch.device, label, notes). Never raises."""
    # Apple Silicon
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple MPS", []

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.version.hip:
            arch = torch.cuda.get_device_properties(0).gcnArchName
            return torch.device("cuda"), f"AMD {name} [{arch}] {vram:.0f}GB", []
        return torch.device("cuda"), f"NVIDIA {name} {vram:.0f}GB", []

    return torch.device("cpu"), f"CPU ({os.cpu_count()} threads)", [
        "No GPU detected — the embed pass will be ~20x slower.",
        "Consider running with --no-serve and leaving it overnight.",
    ]
```

Verify the device actually computes before committing to a long run — a device
that enumerates but faults on the first matmul is a real failure mode on
unsupported AMD cards:

```python
def device_works(dev):
    try:
        a = torch.randn(2048, 2048, device=dev)
        (a @ a).sum().item()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return True
    except Exception:
        return False
```

If this fails, fall back to CPU with a clear warning rather than crashing 8,000
images into the run. Always provide `--device {auto,cuda,mps,cpu}` to override.

### Install matrix

PyTorch ships a different wheel per backend; the user must install the right one
*before* the tool runs. Detect a mismatch and print the exact command.

| Hardware | Install |
|---|---|
| NVIDIA | `uv pip install torch torchvision` (default index bundles CUDA) |
| AMD RDNA2/3, Linux | `uv pip install --index-url https://download.pytorch.org/whl/rocm6.3 torch torchvision` |
| Apple Silicon | `uv pip install torch torchvision` (MPS is built in) |
| No GPU | `uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision` |

The ROCm and CUDA wheels **bundle their runtime** — no system CUDA toolkit, no
system ROCm install, no sudo. For AMD you need only the kernel `amdgpu` driver
and read/write access to `/dev/dri/renderD*` (usually mode `0666`, or add the
user to the `render` group).

Detecting the wrong wheel is worth doing explicitly:

```python
if not torch.cuda.is_available() and has_nvidia_smi():
    warn("nvidia-smi found but this is a CPU-only PyTorch build. "
         "Reinstall: uv pip install torch torchvision")
```

### AMD: unsupported-but-working cards

Consumer Radeons often aren't on AMD's official support list yet work fine with
an architecture override. The prototype ran on an **RX 6700 XT (gfx1031)**:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0   # present gfx1031 as the supported gfx1030
```

- gfx1031/gfx1032 have no Tensile kernel library of their own, so they
  masquerade as **gfx1030** (RX 6800/6900), which does. The override is
  well-trodden and safe.
- **Pin to the `rocm6.3` wheels.** There is a documented regression where this
  override segfaults on gfx1031/gfx1032 under **ROCm 6.4.3+**, with no confirmed
  fix in the 7.2 line. 6.3 sits below it and is verified working.
- `torch.cuda.get_arch_list()` must contain the *target* arch (`gfx1030`). If it
  doesn't, the override cannot help — the wheel has no kernels for it.
- The card will report as `gfx1030` with a generic name ("AMD Radeon Graphics").
  That is the override working, not a fault.
- A `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file` warning is cosmetic —
  it's only device-name lookup failing.

The tool should set `HSA_OVERRIDE_GFX_VERSION` itself when it detects a known
override-able arch and the variable isn't already set:

```python
OVERRIDES = {"gfx1031": "10.3.0", "gfx1032": "10.3.0", "gfx1010": "10.1.0",
             "gfx1012": "10.1.0", "gfx1103": "11.0.0"}
```

This must happen **before** the first CUDA call, so do it at import time or
re-exec. Document it in the log so users can reproduce it by hand.

### Tuning by device

| Device | Batch | DataLoader workers |
|---|---|---|
| CUDA / ROCm | 128 | `cpu_count() - 2` |
| MPS | 64 | 4 |
| CPU | 32 | `cpu_count() - 2` |

Measured on the prototype: batch 128 at 224×224 used **0.8 GB VRAM** and hit
~300 img/s — so the GPU is nowhere near saturated, and **JPEG decode on the CPU
is the bottleneck**. Consequences:

- Don't scale batch size to fill VRAM; it buys nothing. 8 GB cards are fine.
- Spend effort on decode instead: `pillow-simd`, or `torchvision.io.decode_jpeg`
  with `device="cuda"` to decode on the GPU (NVIDIA only).
- Feed the loader enough workers or the GPU starves. `prefetch_factor=4` helped.

Run inference under `torch.no_grad()` and L2-normalise before storing, so search
is a plain dot product later.

---

## 5. Storage layout

**All state lives inside the folder being triaged**, in one dot-directory. No
global database, no config in `$HOME`. Delete the dot-directory and the tool has
left no trace.

```
<target-folder>/
├── holidays/
│   └── IMG_0042.jpg
├── screenshots/
│   └── Screenshot_20250101.png
└── .phototriage/
    ├── index.jsonl          # per-file metadata + perceptual hashes
    ├── embeds.npy           # N×512 float32, L2-normalised
    ├── paths.json           # row id → relative path
    ├── scored.json          # row id → category + confidence
    ├── saved.json           # row ids marked protected
    ├── journal.jsonl        # append-only log of every move
    ├── thumbs/<row>.jpg
    └── quarantine/          # <- moved files land here
        ├── holidays/
        │   └── IMG_0099.jpg
        └── screenshots/
            └── Screenshot_20250101.png
```

### Quarantine design

Quarantine is `.phototriage/quarantine/` **inside the target folder**, and it
**mirrors the original directory structure**.

- **Same filesystem** → moves are `rename()`: instantaneous, atomic. No copying
  a 12 GB pile, no half-written files.
- **Mirrored paths** → restore is unambiguous even for duplicate basenames.
  `holidays/IMG_0042.jpg` and `trip/IMG_0042.jpg` don't collide, so you never
  need id-prefixed filenames to disambiguate.
- **Self-contained** → moving or backing up the folder takes the quarantine and
  all state with it.
- **Recoverable by hand** → it's a normal folder tree. If the tool breaks
  entirely, the user can drag files back in a file manager. Do not invent a
  container format.

Emptying the quarantine is the *only* destructive operation. It must be a
separate explicit command, print a summary, and require confirmation.

**Exclude `.phototriage/` from scanning** or the tool indexes its own thumbnails
and quarantined files.

### `journal.jsonl`

Append-only, one object per operation. This is the undo history and audit trail.

```json
{"op": "quarantine", "ids": [12, 88, 91], "t": 1735689600.0}
{"op": "restore",    "ids": [88],         "t": 1735689642.0}
{"op": "save",       "ids": [7],          "t": 1735689700.0}
```

Rebuild state by replaying the journal at startup. The journal is the source of
truth; never store derived state as authoritative.

---

## 6. Classification

Zero-shot. Each category is several prompt phrasings whose encoded vectors are
averaged and re-normalised; classification is `argmax` cosine similarity,
softmaxed at temperature 100 for a confidence figure.

```python
CATEGORIES = {
    "meme_text": [
        "a meme with large text captions",
        "an image macro with white impact font text",
        "a funny meme picture shared on social media",
        "a picture with a joke written across it",
    ],
    "screenshot": [
        "a screenshot of a phone screen",
        "a screenshot of a chat conversation",
        "a screenshot of a website or an app interface",
        "a screenshot of social media post",
    ],
    "text_graphic": [
        "a poster with a written message",
        "an inspirational quote written on a background",
        "a flyer or advertisement with text",
    ],
    "sticker_art": [
        "a cartoon drawing or illustration",
        "a sticker or clipart graphic",
        "a digital drawing with a plain background",
    ],
    "document": [
        "a scanned document or paperwork",
        "a photograph of a receipt or an invoice",
    ],
    "people_photo": [
        "a candid photograph of people",
        "a selfie of a person",
        "a group photo of friends or family",
    ],
    "event_photo": [
        "a photograph taken at a party or celebration",
        "a photo from a wedding or a birthday",
    ],
    "place_photo": [
        "a photograph of a landscape or scenery",
        "a travel photo of a place or a building",
    ],
    "food_photo":   ["a photograph of a meal or food on a plate"],
    "animal_photo": ["a photograph of a pet dog or cat", "a photo of an animal"],
    "product":      ["a product photo for online shopping",
                     "a catalogue photo of an item for sale"],
}
```

Group for the UI as **JUNK** (`meme_text`, `screenshot`, `text_graphic`,
`sticker_art`), **REVIEW** (`product`, `document`) and **KEEP** (the rest).

`product` and `document` are deliberately *not* junk — see §9.1. No category
should ever be bulk-deleted without the user having looked at it.

Because embeddings are cached, re-running classification with better prompts
costs seconds. Make prompts user-editable in a config file and expose a
"re-classify" button. Treat the prompt set as tunable, not fixed.

---

## 7. The UI

Single page. Dark. Sticky filter header, thumbnail grid, status bar. No
framework.

```
┌────────────────────────────────────────────────────────────────────┐
│ [ search: "birthday cake"          ] [group▾][cat▾][folder▾][show▾]│
├────────────────────────────────────────────────────────────────────┤
│ 1,022 match · showing 300 · 7,244 on disk · 96 saved               │
│        [Select shown A][Select all matching][Save S][Delete D]     │
├────────────────────────────────────────────────────────────────────┤
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                         │
│  │meme│ │scrn│ │ ✓  │ │peop│ │food│ │meme│    ← click to select    │
│  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘      shift+click range  │
└────────────────────────────────────────────────────────────────────┘
```

Each tile shows a colour-coded category badge (JUNK / REVIEW / KEEP), the
similarity score when searching, confidence, dimensions and source folder.

### Keyboard shortcuts

Keep these identical to the prototype.

| Key | Action |
|---|---|
| `A` | select everything currently shown |
| `D` | quarantine the selection (confirm dialog) |
| `S` | **save/protect** — never deletable, drops out of normal results |
| `C` | clear selection |
| `U` | undo the last quarantine batch |
| `R` | restore selection (quarantine view only) |
| `F` | find visually similar to the tile under the cursor |
| `Esc` | close the lightbox |
| `Enter` | run the search (from the search box) |

Mouse: click toggles, **shift+click** selects a range, **double-click** opens a
full-size lightbox. `S` and `F` act on the tile under the cursor when nothing is
selected, so you can sweep through hovering and tapping.

Ignore keystrokes when focus is in an `<input>` or `<select>`, and ignore them
when a modifier is held, or `Ctrl+A` will nuke the selection semantics.

### The save/protect mechanic

`S` is what makes long triage sessions converge. A saved image is:

- **protected** — the delete endpoint refuses it, **server-side**, not just in
  the UI;
- **hidden** from default results.

So the pile still to be judged shrinks with every pass. A `show` filter toggles
*hide saved* (default) / *only saved* / *everything*. Persist to `saved.json`
and reload at startup.

### Search behaviour

`GET /api/search?q=&cat=&folder=&group=&show=&like=&limit=&offset=`

- `q` — encode with the CLIP text encoder, rank candidates by cosine.
  **Rank, don't threshold.** There is no meaningful universal cutoff; a sorted
  top-N is more useful than an arbitrary similarity floor.
- `like=<id>` — same, but against an image embedding.
- Cache encoded query vectors; users retype the same terms constantly.

Search works cross-language: the prototype found Spanish-language chess memes
from the English query `chess meme`, because it matches the *picture*.

Good queries to suggest in the placeholder: `birthday cake`, `receipt`,
`my dog`, `people at the beach`, `chat screenshot`.

### Review mode (do not skip this)

A grid of 300 thumbnails is for *acting*. Before any bulk action on a whole
category, the user needs a mode for *judging*: every image in that category,
sorted most-confident first, at a size where mistakes are visible.

The prototype shipped this as a static self-contained HTML contact sheet with
base64 thumbnails, and it is what caught the model's false positives. Build it
in as a first-class view, and consider requiring a category be viewed at least
once before it can be bulk-deleted.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/search` | filtered + ranked page of results |
| GET | `/api/ids` | **every** id matching current filters (powers "select all matching") |
| GET | `/api/stats` | per-category counts |
| GET | `/thumb/<id>` | cached thumbnail |
| GET | `/full/<id>` | original file (also serves from quarantine) |
| POST | `/api/quarantine` | move into the mirrored quarantine tree |
| POST | `/api/restore` | move back — must work for **any** batch, not just the last |
| POST | `/api/save` | protect / unprotect |

`/api/ids` and `/api/search` **must share one filter function**. If they drift,
the count shown stops matching what "select all" acts on — which is exactly how
someone deletes more than they meant to.

---

## 8. CLI

```
photo-triage <folder>                # scan, embed, classify, thumbnail, serve
photo-triage <folder> --no-serve     # build caches only
photo-triage <folder> --reclassify   # re-run stage 3 with edited prompts
photo-triage <folder> --status       # counts: indexed, quarantined, saved
photo-triage <folder> --dedupe       # report exact duplicates (dry run default)
photo-triage purge <folder>          # permanently empty quarantine (confirm!)
photo-triage restore-all <folder>
```

Default `<folder>` to cwd. Every stage resumable and idempotent: re-running
after adding photos embeds only the new ones. Key the cache on
`(relative path, size, mtime)`.

---

## 9. Pitfalls (read this before writing code)

Every one of these was hit in the prototype. They are the actual product
requirements.

### 9.1 The model is confidently wrong in specific, predictable ways

Sampled from real output, with confidence scores:

| Image | Classified | Confidence | Reality |
|---|---|---|---|
| A photo of a bed in a bedroom | `product` | 0.92 | Someone's own room |
| A woman posing beside a cartoon mural | `sticker_art` | **0.99** | Real photo at an art installation |
| A Spanish chess meme | `meme_text` | 0.98 | Correct |
| A VS Code window | `screenshot` | 0.99 | Correct |

`meme_text` and `screenshot` are reliable. **`product` is not a junk category** —
CLIP maps "photograph containing an object" onto it, which covers most indoor
personal photos. `sticker_art` fires on any photo that *contains* flat artwork.

**Consequence: high confidence is not correctness.** Never let confidence alone
authorise deletion. In the prototype's library scan, 39 of 260 flagged
screenshots scored below 0.5 — the tail is where the errors live, but the head
is where the *dangerous* errors live, because that's what a user trusts.

### 9.2 Never bulk-delete without visual review

The prototype's UI made "Select all matching" easy and deletion one keystroke.
The user deleted **21,488 of 23,584 images (91%)**, including **1,617 of 2,454**
people photos and **1,209 of 1,239** animal photos. Two random samples pulled
from the bin afterwards were both obviously wanted — a child in a football kit,
three friends laughing at a party.

Mitigations the rebuild must have:

- A review mode (§7) that renders a whole category **before** any bulk action.
- "Select all matching" must state the count in a confirm dialog and look
  visually distinct from "select shown". These two actions differing silently —
  300 loaded vs 10,000 matching — is the single most dangerous thing in the UI.
- Deletion rates above ~70% of a non-junk category should trigger a warning.
- Quarantine, never delete. Purging is a separate deliberate command.

### 9.3 Undo must handle any batch, not just the last

Single-level undo is not enough for a session with forty selections. Implement
restore over arbitrary ids from the journal, plus a browsable quarantine view
with its own filters and search — that's what `R` is for. The user should be
able to semantically search the bin (`"photos of people"`) to audit what they
threw away.

### 9.4 Duplicate detection: choose the right hash

- **Exact duplicates** — md5 of the **decoded pixel buffer**, not file bytes.
  Two files can be pixel-identical yet differ on disk (different EXIF, different
  encoder). Comparing file checksums finds nothing.
- **Near duplicates** — perceptual hash (`phash`). Catches recompressed and
  resized re-sends.
- Keeper rule for a group: **earliest folder/date → largest file → shortest
  name.** Largest is usually least re-compressed.

In the prototype, exact-pixel dedup over 6,979 images found 52 redundant copies
(9 MB) — almost all `received`/`sent` pairs of the same image, i.e. someone
forwarding back a photo they'd been sent. Perceptual dedup found more but needs
confirmation, because near-identical is not identical: burst shots and
slightly-different crops are legitimately separate photos.

Default `--dedupe` to a **dry run** that prints the plan.

### 9.5 Re-sends: the same photo arrives twice, years apart

A photo taken in 2019 and forwarded in 2026 arrives as a fresh file with a fresh
name. Nothing in the file marks it as a re-send.

Detect by grouping on perceptual hash and flagging groups whose folder dates
differ by more than ~6 months. The prototype found 8 such pairs, two re-sent on
the *exact anniversary* of the original. In every case the older copy was equal
or better quality — one 16.2 Mpx original had been re-sent back as 1.9 Mpx.

This is worth surfacing because the **newer copy is the worse one**, which is
the opposite of what a naive "keep the newest" rule would do. It only works when
both copies survive; say so in the UI rather than implying the scan is complete.

### 9.6 Files lie about what they are

- Files with a `.jpg` extension that are actually **WebP** or PNG. Sniff the
  real type from magic bytes; never trust the extension.
- Truncated and corrupt JPEGs exist in real exports — the prototype found 28
  that no decoder could read. Handle failure **per file**; one bad image must
  never kill a batch of 20,000. Surface them as an `unreadable` bucket rather
  than silently dropping them.
- Animated GIFs and multi-frame images: seek to frame 0 before embedding.

### 9.7 Verify before you destroy

The prototype tried to archive a folder and delete the source in one shell
command. The `find` output contained spaces, `tar` got a broken argument list
and wrote an empty archive — and the `rm -rf` ran anyway. The data was lost.

**Never chain create-artifact and delete-source in one unchecked step.** Verify
the artifact exists, is non-empty, and has the expected entry count first. This
applies to `purge` and to any export feature.

### 9.8 Operational notes

- `Image.MAX_IMAGE_PIXELS = None`, or Pillow raises on large panoramas.
- Paths contain spaces, accents and emoji. Never build shell commands by string
  concatenation — use `subprocess` argument lists.
- Long work goes in a background thread with progress the UI can poll; a
  20-minute embed pass must not look frozen.
- Test the GPU with a real matmul before a long run (§4a). A device that
  enumerates but faults on first compute is a genuine failure mode on
  unsupported AMD cards, and you want to know at second 2, not minute 40.
- Buffer-aware logging: piping a progress bar through `tail` hides it entirely.
- `pkill -f <pattern>` matches the shell that ran it. Kill by port or PID file.
- Serve on `127.0.0.1` only. This is personal photo data; never bind `0.0.0.0`.

---

## 10. Suggested repo layout

```
photo-triage/
├── pyproject.toml
├── README.md
├── PROJECT.md               # this file
├── photo_triage/
│   ├── __main__.py          # CLI entry
│   ├── scan.py              # stage 1
│   ├── embed.py             # stage 2  (device detection lives here)
│   ├── classify.py          # stage 3
│   ├── thumbs.py            # stage 4
│   ├── server.py            # stage 5, Flask
│   ├── quarantine.py        # move/restore/purge + journal
│   ├── dedupe.py            # exact + perceptual
│   ├── review.py            # contact-sheet generator
│   ├── prompts.py           # default category prompts
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
└── tests/
    ├── test_quarantine.py
    ├── test_dedupe.py
    └── test_filters.py
```

### Tests worth having on day one

- Quarantine a file in a nested folder → restore → **byte-identical, same path**.
- Two files with the same basename in different folders → both quarantined →
  both restore correctly.
- Restore an arbitrary earlier batch, not just the most recent.
- A saved file cannot be deleted, enforced **server-side** (POST directly, not
  through the UI).
- `/api/ids` and `/api/search` return consistent counts for identical filters.
- Exact dedup treats two pixel-identical files with different EXIF as duplicates.
- Interrupting the embed stage and re-running resumes rather than restarts.
- `.phototriage/` is never itself indexed.

---

## 11. Roadmap

Roughly in value order.

1. **Face clustering** (`insightface`) — "photo I care about" correlates with
   "contains people I know" far better than any CLIP category. Cluster faces,
   let the user name a few clusters, filter by person. Probably the single
   biggest improvement available.
2. **Interactive learning** — the user's save/delete decisions are labels. Train
   a logistic-regression head on the cached embeddings; it personalises fast and
   costs nothing, since the embeddings already exist. This turns a generic
   classifier into *your* classifier after a few hundred decisions.
3. **OCR text density** (`easyocr`) as a *secondary* meme signal. High text
   coverage plus flat colour is a strong junk indicator; slow, so run it only on
   ambiguous cases.
4. **Blur / quality detection** — variance of Laplacian. Accidental pocket shots
   and out-of-focus frames are junk no semantic model will catch.
5. ~~**Video support**~~ — done. Frames are sampled about every two seconds
   and each keeps its own vector; a clip scores as its best moment rather than
   its average, because averaging a multi-scene clip points the result at
   nothing that is in it. Audio is still not read.
6. **Bigger model option** — SigLIP or ViT-L/14 for users with VRAM to spare.

---

## 12. Licence

Nothing here depends on anything non-permissive: `open_clip` (MIT), the LAION-2B
weights, PyTorch (BSD), Pillow (MIT-CMU), Flask (BSD), `imagehash` (BSD). MIT is
a reasonable default for the project itself.
