# Arch packaging

`PKGBUILD` and `.SRCINFO` for the AUR package
[photo-triage](https://aur.archlinux.org/packages/photo-triage).

## Build it locally

```bash
cd packaging
makepkg -si
```

## Publish or update the AUR package

The AUR is a separate git repository holding only `PKGBUILD` and `.SRCINFO`.
It needs an AUR account with your SSH public key added under
[Account settings](https://aur.archlinux.org/account/).

First time:

```bash
git clone ssh://aur@aur.archlinux.org/photo-triage.git aur-photo-triage
cp packaging/PKGBUILD packaging/.SRCINFO aur-photo-triage/
cd aur-photo-triage
git add PKGBUILD .SRCINFO
git commit -m "Initial import: photo-triage 0.1.0"
git push
```

For each new release, tag and publish upstream first, since `sha256sums` is
pinned to the tarball GitHub generates from the tag:

```bash
git tag -a v0.2.0 -m "..." && git push origin v0.2.0
gh release create v0.2.0 --notes "..."

cd packaging
sed -i 's/^pkgver=.*/pkgver=0.2.0/; s/^pkgrel=.*/pkgrel=1/' PKGBUILD
updpkgsums                       # re-downloads the tarball and pins its hash
makepkg --printsrcinfo > .SRCINFO
makepkg -f                       # never push a PKGBUILD you have not built
```

Then copy both files into the AUR clone and push, as above.

## Why PyTorch is an optdepend

Browsing, filtering, searching by category, quarantining and restoring an
already-triaged folder all work without it, so a hard dependency would make
every user download two gigabytes to look at a folder they had already
embedded elsewhere.

Which PyTorch is also not ours to choose. `python-pytorch-rocm` and
`python-pytorch-cuda` are mutually exclusive and depend on hardware only the
user can see. photo-triage detects what is present at runtime and, on a
distribution-managed Python, prints the package to install rather than
installing anything itself.

`python-imagehash` and `python-open-clip-torch` come from the AUR rather than
the official repositories, so an AUR helper will pick them up but a plain
`makepkg -si` will ask you to install `python-imagehash` first.

## A note on the tarball size

The source tarball is about 12 MB, most of which is `docs/demo.mp4` and
`docs/demo.gif`. That is downloaded once per build. If it becomes annoying,
move the demo files to release assets and drop them from the tree.
