# photo-triage

## Documents

- [PROJECT.md](PROJECT.md) — architecture, endpoints, data model. Owns *behaviour*.
- [DESIGN.md](DESIGN.md) — UI spec. Owns *appearance*.
- [STYLE.md](STYLE.md) — coding style, from Ousterhout's *A Philosophy of
  Software Design*. Owns *how the code is shaped*.

## Rules for writing code here

**Read [STYLE.md](STYLE.md) before writing or reviewing code.** It is not
advisory. In particular:

1. **Write the interface comment first**, before the implementation. If it comes
   out long or awkward, the abstraction is wrong — fix the design, not the prose.
2. **Design it twice.** For any new module boundary, cache format or interaction
   model, sketch two genuinely different approaches and say why you picked one.
3. **Deep modules.** Simple interface, substantial implementation. Prefer fewer
   larger modules to many small ones. No pass-through methods.
4. **One design decision, one module.** File formats, path-mapping rules and
   thresholds each live in exactly one place.
5. **Pull complexity downward.** Compute defaults rather than adding config
   knobs. A new parameter needs a justification.
6. **Define errors out of existence** before handling them. Aggregate the rest
   at one high-level handler per stage.
7. **Run the STYLE.md §16 red-flag checklist** before declaring work done, and
   report anything you knowingly left in.

Do not narrow scope silently. If a design decision here conflicts with
PROJECT.md, raise it rather than quietly deviating.
