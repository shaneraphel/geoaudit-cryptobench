#!/usr/bin/env python3.12
"""Draw what the word "resolution" spans across two experiments labelled EM.

The point of the figure
-----------------------
One panel is the ERa PROTAC ternary complex at 4.3 A by single-particle cryo-EM.
The other is proteinase K at 1.09 A by electron crystallography. Both are
returned by a query for ``experimental_method == "EM"``. Placed at a matched
physical scale, the pair says two things that the ESR1 appendix asserts in prose
and that a reader is entitled to see: that "EM" is not one measurement, and that
at 4.3 A there is no atom to point at, so a claim about how a degrader sits in a
pocket cannot be read off this density.

That second reading is the reason the figure is in a repository whose detector
never touches a density map. The appendix declines to assert a binding pose. This
is what declining looks like when the experimental map is in front of you.

Labels are read from results/external/EMDB_MAPS.json rather than typed, so the
method and the resolution under each panel cannot drift from the artifact that
asserted them. Following the convention of make_official_figures.py, no title is
burned into the pixels: the description belongs in the caption where it can be
typeset and corrected.

Orientation
-----------
MRC stores the fastest-varying axis as columns and records which physical axis
each of columns, rows and sections is, in MAPC/MAPR/MAPS. EMD-55233 is 1/2/3 and
EMD-46871 is 3/2/1, so the second volume is transposed on read. Without that the
two panels would be sliced along different physical directions while the caption
claimed they were comparable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pocket_bench.paths import ROOT  # noqa: E402

PROV = ROOT / "results/external/EMDB_MAPS.json"
FIGDIR = ROOT / "figures"
# The physical width of the matched-scale crop. Chosen because it is a little
# wider than an aromatic ring system plus its first shell, i.e. the scale at
# which a claim about a ligand pose would have to be made.
ZOOM_ANGSTROM = 24.0
MODE_DTYPE = {0: "<i1", 1: "<i2", 2: "<f4", 6: "<u2", 12: "<f2"}


def read_mrc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the volume indexed by physical (X, Y, Z) and the voxel size in A."""
    with gzip.open(path, "rb") as fh:
        h = fh.read(1024)
        ncol, nrow, nsec, mode = struct.unpack_from("<4i", h, 0)
        mx, my, mz = struct.unpack_from("<3i", h, 28)
        ax, ay, az = struct.unpack_from("<3f", h, 40)
        mapc, mapr, maps_ = struct.unpack_from("<3i", h, 64)
        nsymbt = struct.unpack_from("<i", h, 92)[0]
        if nsymbt:
            fh.read(nsymbt)
        if mode not in MODE_DTYPE:
            raise ValueError(f"{path.name}: unsupported MRC mode {mode}")
        buf = fh.read()
    want = ncol * nrow * nsec
    arr = np.frombuffer(buf, dtype=MODE_DTYPE[mode], count=want)
    # C order over (sections, rows, columns).
    arr = arr.reshape(nsec, nrow, ncol).astype(np.float32)

    # arr axes (sec, row, col) carry physical axes (maps_, mapr, mapc). Permute to
    # (X, Y, Z); a wrong permutation would slice the two maps along different
    # directions while the caption claims they are comparable.
    phys = {maps_: 0, mapr: 1, mapc: 2}
    if sorted(phys) != [1, 2, 3]:
        raise ValueError(f"{path.name}: MAPC/MAPR/MAPS not a permutation of 1,2,3")
    vol = np.transpose(arr, axes=[phys[1], phys[2], phys[3]])
    vox = np.array([ax / mx, ay / my, az / mz], dtype=np.float64)
    return vol, vox


def densest_slice(vol: np.ndarray, thresh_sigma: float = 3.0) -> int:
    """Index along Z of the slice carrying the most above-threshold signal.

    A central slice of a 248 A box is mostly solvent for a particle that occupies
    part of it, which would show an empty panel and say nothing. Counting voxels
    above the map's own noise scale finds where the molecule is without assuming
    it sits at the centre.
    """
    mu, sd = float(vol.mean()), float(vol.std())
    if sd <= 0:
        return vol.shape[2] // 2
    hot = (vol > mu + thresh_sigma * sd).sum(axis=(0, 1))
    return int(np.argmax(hot))


def crop_to_signal(sl: np.ndarray, vox_xy, pad_a: float = 8.0,
                   thresh_sigma: float = 2.0):
    """Bounding box of above-noise density in a slice, padded, as index slices."""
    mu, sd = float(sl.mean()), float(sl.std())
    hot = sl > mu + thresh_sigma * sd
    if not hot.any():
        return slice(None), slice(None)
    out = []
    for axis, vx in zip((1, 0), (vox_xy[0], vox_xy[1])):
        idx = np.where(hot.any(axis=axis))[0]
        pad = int(round(pad_a / vx))
        out.append(slice(max(0, int(idx[0]) - pad),
                         min(sl.shape[1 - axis], int(idx[-1]) + 1 + pad)))
    return out[0], out[1]


def zoom_window(sl: np.ndarray, vox_xy, width_a: float):
    """A width_a x width_a crop centred on the brightest voxel of the slice."""
    ny, nx = sl.shape
    cy, cx = np.unravel_index(int(np.argmax(sl)), sl.shape)
    hy, hx = int(round(width_a / vox_xy[1] / 2)), int(round(width_a / vox_xy[0] / 2))
    y0, x0 = max(0, cy - hy), max(0, cx - hx)
    return sl[y0:min(ny, y0 + 2 * hy), x0:min(nx, x0 + 2 * hx)]


def show(ax, img, vox_xy, bar_a: float, label: str, span_a: float | None = None) -> None:
    lo, hi = np.percentile(img, [2.0, 99.8])
    w, h = img.shape[1] * vox_xy[0], img.shape[0] * vox_xy[1]
    ax.imshow(img, cmap="bone", vmin=lo, vmax=hi, origin="lower",
              interpolation="nearest", extent=[0, w, 0, h])
    if span_a is not None:
        # Both panels of a row get the same physical span, so their apparent sizes
        # are comparable. Left unequal, two crops of 92 A and 61 A would occupy the
        # same area on the page and invite a comparison the scale bars deny.
        ax.set_xlim((w - span_a) / 2, (w + span_a) / 2)
        ax.set_ylim((h - span_a) / 2, (h + span_a) / 2)
        ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.55")
    # Anchor the bar in the visible window, which is not the whole image once a
    # common span has been imposed.
    (vx0, vx1), (vy0, vy1) = ax.get_xlim(), ax.get_ylim()
    vw, vh = vx1 - vx0, vy1 - vy0
    x0 = vx1 - bar_a - 0.06 * vw
    y0 = vy0 + 0.07 * vh
    ax.plot([x0, x0 + bar_a], [y0, y0], color="w", lw=3, solid_capstyle="butt")
    ax.text(x0 + bar_a / 2, y0 + 0.035 * vh, f"{bar_a:g} $\\AA$",
            color="w", ha="center", va="bottom", fontsize=8)
    ax.set_title(label, fontsize=9, pad=4)


def maps_present(prov: dict | None = None) -> bool:
    """Whether every declared volume is on disk.

    The volumes are 139 MB and gitignored, so a fresh checkout has the committed
    image but not the data behind it. ``make_official_figures.py`` uses this to
    decide between redrawing and keeping the committed bytes, rather than
    silently dropping the figure out of the provenance record.
    """
    if prov is None:
        if not PROV.exists():
            return False
        prov = json.loads(PROV.read_text())
    return all((ROOT / r["path"]).exists() for r in prov["maps"])


def render(out: Path) -> Path:
    """Draw the figure and return where it went."""
    return _draw(json.loads(PROV.read_text()), out)


def caption() -> str:
    """The caption, built from the artifact rather than typed.

    Every number and every method name in it is read from EMDB_MAPS.json, so the
    sentence under the image cannot claim a method the assertion did not confirm.
    """
    prov = json.loads(PROV.read_text())
    by = {r["emdb_metadata"]["method"]: r for r in prov["maps"]}
    sp, ec = by["singleParticle"], by["electronCrystallography"]
    # No backticks and no straight double quotes: this string is spliced into both
    # README.md and a LaTeX macro, and in LaTeX both characters come out as the
    # wrong glyph. Every other caption in this generator observes the same rule.
    return (
        f"How far the word resolution stretches across two entries that a "
        f"single RCSB query returns together, both carrying the experimental "
        f"method EM. Left, "
        f"{sp['emd_id']}: single-particle cryo-EM at "
        f"{sp['emdb_metadata']['resolution_angstrom']:g} A, a VHL-recruiting "
        f"PROTAC ternary complex containing ERa. Right, {ec['emd_id']}: "
        f"electron crystallography (MicroED) at "
        f"{ec['emdb_metadata']['resolution_angstrom']:g} A, proteinase K -- not "
        f"cryo-EM, and included because it is the entry this repository first "
        f"mistook for one by ordering candidates on resolution. Top row, the "
        f"densest slice of each map at a common physical scale; bottom row, "
        f"{ZOOM_ANGSTROM:g} A across each, the scale at which a claim about how "
        f"a ligand sits would have to be made. At "
        f"{sp['emdb_metadata']['resolution_angstrom']:g} A there is no atom to "
        f"point at, which is why the ESR1 appendix declines to assert a binding "
        f"pose. No number in the benchmark reads a density map; the detector "
        f"consumes apo coordinates. Volumes are gitignored, and "
        f"results/external/EMDB_MAPS.json carries the URL, the asserted method "
        f"and the sha256 of each."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=FIGDIR / "fig_emdb_resolution.png")
    a = ap.parse_args(argv)
    out = _draw(json.loads(PROV.read_text()), a.out)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


def _draw(prov: dict, out: Path) -> Path:
    # Single-particle first, so the reading order is "the one we care about, then
    # the one that shows what it is missing".
    rows = sorted(prov["maps"], key=lambda r: not r["is_single_particle_cryo_em"])

    panels = []
    for r in rows:
        path = ROOT / r["path"]
        if not path.exists():
            raise SystemExit(f"{r['emd_id']}: {path} absent; run tools/emdb_maps.py "
                             f"--fetch --write first")
        vol, vox = read_mrc(path)
        sl = vol[:, :, densest_slice(vol)].T
        ys, xs = crop_to_signal(sl, vox[:2])
        full = sl[ys, xs]
        panels.append({
            "row": r, "vox": vox, "full": full,
            "zoom": zoom_window(sl, vox[:2], ZOOM_ANGSTROM),
            "span": max(full.shape[1] * vox[0], full.shape[0] * vox[1]),
        })
    top_span = max(p["span"] for p in panels)

    fig, axes = plt.subplots(2, len(panels), figsize=(4.1 * len(panels), 8.4))
    for col, p in enumerate(panels):
        r, vox, meta = p["row"], p["vox"], p["row"]["emdb_metadata"]
        kind = ("single-particle cryo-EM" if r["is_single_particle_cryo_em"]
                else "electron crystallography (MicroED)")
        show(axes[0, col], p["full"], vox[:2], 50.0,
             f"{r['emd_id']}  ·  {kind}\n{meta['resolution_angstrom']:g} $\\AA$ "
             f"({meta['resolution_type'].lower()})  ·  voxel {vox[0]:.3g} $\\AA$",
             span_a=top_span)
        show(axes[1, col], p["zoom"], vox[:2], 5.0,
             f"{ZOOM_ANGSTROM:g} $\\AA$ across\n"
             f"EMDB method field: {meta['method']}",
             span_a=ZOOM_ANGSTROM)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
