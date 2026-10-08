#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composes Figure 11 (Representative Structural Snapshots) from the PyMOL
renders in figures/renders/ into a single publication-style multi-panel
PNG/PDF, matching the font/style conventions of the project's other figures.
Does not re-render anything; purely a matplotlib layout step.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg


def autocrop_white(img, pad=15, white_thresh=0.98):
    """Trims uniform near-white margins from a ray-traced PNG (RGB/RGBA
    float array) so the molecule fills more of its panel, with a small
    pixel pad kept around the detected content bounding box."""
    rgb = img[:, :, :3]
    mask = (rgb < white_thresh).any(axis=2)
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, img.shape[0] - 1)
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, img.shape[1] - 1)
    return img[r0:r1 + 1, c0:c1 + 1]

BASE_DIR = Path("/home/luiz/SIMULACOES/her-oer-co2rr-mofs")
RENDER_DIR = BASE_DIR / "figures" / "renders"
FIG_DIR = BASE_DIR / "figures"

mpl.rcParams.update({
    # dvipng (needed for text.usetex) isn't available in this environment;
    # matplotlib's built-in mathtext renders the same $...$ math fine without it.
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})

fig = plt.figure(figsize=(12.0, 14.0))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.07], hspace=0.35, wspace=0.08)

panels = [
    ("oer_champion_CoMOF74_framework_overview.png", "(a) OER: Co-MOF-74 channel"),
    ("her_champion_CuMOF74_framework_overview.png", "(b) HER: Cu-MOF-74 channel"),
    ("co2rr_champion_qmofdc7e5a3_framework_overview.png", "(c) CO$_2$RR champion channel"),
    ("oer_champion_CoMOF74_active_site.png", r"(d) OER site: $*$OH/Co"),
    ("her_champion_CuMOF74_active_site.png", r"(e) HER site: $*$H/Cu"),
    ("co2rr_champion_qmofdc7e5a3_active_site.png", r"(f) CO$_2$RR site: $*$COOH/Co"),
]

for idx, (fname, caption) in enumerate(panels):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])
    img = autocrop_white(mpimg.imread(str(RENDER_DIR / fname)))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=9.5, pad=4)

ax_g = fig.add_subplot(gs[2, 1])
img_g = autocrop_white(mpimg.imread(str(RENDER_DIR / "solvation_example_CoMOF74_OH_2H2O_active_site.png")))
ax_g.imshow(img_g)
ax_g.axis("off")
ax_g.set_title(r"(g) Solvation: $*$OH + $n_{\mathrm{H_2O}}=2$", fontsize=9.5, pad=4)

ax_legend = fig.add_subplot(gs[3, :])
ax_legend.axis("off")
ax_legend.text(0.5, 0.5,
               "Grey: framework (C/O/H, metal centres as large spheres). Coloured: chemisorbed intermediate "
               "(d-f, g) or explicit pore water (g, blue). Renders: PyMOL 3.1.0, ray-traced from "
               "MACE-MP-0-relaxed structures. (a-c) show a short pore-channel segment (periodic repeat "
               "along the short cell axis); (d-f, g) show a local cutoff sphere ($\\sim$7 Å) around the "
               "adsorption site, so some framework bonds are deliberately cut at the cluster boundary.",
               ha="center", va="center", fontsize=8.5, wrap=True, transform=ax_legend.transAxes)

plt.tight_layout()
png_path = FIG_DIR / "fig11_representative_structures.png"
pdf_path = FIG_DIR / "fig11_representative_structures.pdf"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()
print(f"Saved {png_path} and {pdf_path}")
