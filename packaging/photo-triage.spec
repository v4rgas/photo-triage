# PyInstaller spec for the standalone build.
#
# Deliberately excludes PyTorch, torchvision and open_clip. Baking one of them
# in would mean choosing an accelerator on the user's behalf, and the choice is
# not ours: a CUDA build is 2.5 GB and useless on a Radeon, a CPU build runs
# twenty times slower than the hardware most people have. So the binary ships
# everything else, detects the GPU on first run and installs the wheel that
# matches it into a cache directory beside itself (see runtime.py).
#
# pip is bundled on purpose, because that first-run install has no interpreter
# on PATH to shell out to and drives pip in-process instead.
#
#   pyinstaller packaging/photo-triage.spec

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden = [
    *collect_submodules("pip"),
    *collect_submodules("photo_triage"),
    "imagehash",
    "scipy.fftpack",       # what imagehash's phash actually uses
    "PIL._tkinter_finder",
]

datas = [
    # The UI is read off disk at request time, so it has to travel with us.
    ("../photo_triage/static", "photo_triage/static"),
    *collect_data_files("pip"),
    *collect_data_files("av"),
]

analysis = Analysis(
    ["launcher.py"],
    pathex=[".."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    excludes=[
        # The whole point of the slim build.
        "torch", "torchvision", "open_clip", "open_clip_torch",
        # Pulled in transitively by scipy and friends, never used here.
        "matplotlib", "tkinter", "IPython", "pytest", "setuptools._distutils",
    ],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="photo-triage",
    console=True,
    strip=False,
    upx=False,          # upx mangles some shared objects and saves little here
    onefile=True,
)
