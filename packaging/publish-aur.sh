#!/usr/bin/env bash
# Push packaging/PKGBUILD and packaging/.SRCINFO to the AUR.
#
# The AUR repository holds only those two files. Everything else about the
# project lives on GitHub, and the PKGBUILD points at a release tarball there,
# so publishing here is the last step rather than the first.
#
# Requires an AUR account with your SSH public key registered at
# https://aur.archlinux.org/account/

set -euo pipefail

pkgname=photo-triage
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cd "$here"

# .SRCINFO is generated, and an out-of-date one is the most common way an AUR
# package ends up describing something other than what it builds.
makepkg --printsrcinfo > .SRCINFO
if ! git -C "$here" diff --quiet -- .SRCINFO 2>/dev/null; then
  echo "note: .SRCINFO was stale and has been regenerated; commit it upstream too"
fi

version=$(awk -F= '/^pkgver=/{print $2}' PKGBUILD)
release=$(awk -F= '/^pkgrel=/{print $2}' PKGBUILD)
echo "publishing $pkgname $version-$release"

# Refuse to publish a PKGBUILD whose source hash has not been pinned, since
# 'SKIP' would let the tarball change under everyone who installs it.
if grep -q "sha256sums=('SKIP')" PKGBUILD; then
  echo "error: sha256sums is still SKIP. Run updpkgsums first." >&2
  exit 1
fi

git clone "ssh://aur@aur.archlinux.org/$pkgname.git" "$work/aur"
cp PKGBUILD .SRCINFO "$work/aur/"
cd "$work/aur"

if git diff --quiet; then
  echo "nothing to publish: the AUR already has this exact PKGBUILD"
  exit 0
fi

git add PKGBUILD .SRCINFO
git commit -m "$pkgname $version-$release"
git push
echo "published: https://aur.archlinux.org/packages/$pkgname"
