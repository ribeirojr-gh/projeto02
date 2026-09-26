#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 09: Reaction Coordinate Free Energy Diagrams with Atomistic Active-Site Insets
Project: mofs-mace-her-oer-co2rr (Repo: projeto02-26092026)

Generates canonical electrocatalysis Gibbs free energy profiles at varying potentials
with rendered atomistic insets of the active node and adsorbates:
  1. Fig 4: OER Free Energy Profile (*, *OH, *O, *OOH, O2) at U = 0 V, 1.23 V, and U_L
  2. Fig 5: HER Free Energy Profile (*, *H, 1/2 H2) at U = 0 V and U = -η_HER
  3. Fig 6: CO2RR Free Energy Profile (CO2, *COOH, *CO, CO) at U = 0 V and U = U_L

Dependencies:
    pip install matplotlib ase pandas numpy
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ase.io
from ase.visualize.plot import plot_atoms

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STRUCTURES_DIR = BASE_DIR / "structures" / "mace_relaxed"
PRISTINE_DIR = BASE_DIR / "structures" / "pristine_mofs"
FIG_DIR = BASE_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"

for d in [FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHE_SUMMARY_CSV = DATA_DIR / "che_electrocatalysis_summary.csv"
SITES_CSV = DATA_DIR / "active_sites_summary.csv"
LOG_FILE = LOG_DIR / "results.log"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
    "lines.linewidth": 2.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("script_09_free_energy_diagrams")


def extract_active_cluster(cif_path: Path, site_idx: int, radius: float = 3.6):
    """Extracts a localized cluster around the active metal site for rendering."""
    atoms = ase.io.read(str(cif_path))
    if site_idx >= len(atoms):
        site_idx = 0
    dists = atoms.get_distances(site_idx, range(len(atoms)), mic=True)
    cluster_idx = [i for i, d in enumerate(dists) if d <= radius]
    # Ensure adsorbate atoms (at end of atoms object) are included
    for i in range(max(0, len(atoms)-4), len(atoms)):
        if i not in cluster_idx:
            cluster_idx.append(i)
    cluster = atoms[cluster_idx]
    cluster.center()
    return cluster


def draw_stepped_profile(ax, levels, color, label, linestyle="-", step_width=0.3):
    """Draws stepped horizontal energy levels connected by dashed lines."""
    n = len(levels)
    for i, val in enumerate(levels):
        ax.plot([i - step_width, i + step_width], [val, val], color=color, linestyle=linestyle, linewidth=2.5)
        if i < n - 1:
            next_val = levels[i + 1]
            ax.plot([i + step_width, i + 1 - step_width], [val, next_val], color=color, linestyle=":", alpha=0.7, linewidth=1.4)
    # Dummy plot for legend
    ax.plot([], [], color=color, linestyle=linestyle, label=label)


def plot_oer_diagram(q_id: str = "qmof-73ded45"):
    """Figure 4: OER Free Energy Diagram at 3 potentials with atomistic insets."""
    df = pd.read_csv(CHE_SUMMARY_CSV).set_index("qmof_id")
    if q_id not in df.index:
        q_id = df.index[0]
    row = df.loc[q_id]

    dg_oh = row["delta_G_OH_eV"]
    dg_o = row["delta_G_O_eV"]
    dg_ooh = row["delta_G_OOH_eV"]
    eta = row["eta_OER_V"]
    u_l = 1.23 + eta

    # Levels at U = 0 V
    g_u0 = [0.0, dg_oh, dg_o, dg_ooh, 4.92]
    # Levels at U = 1.23 V
    g_u123 = [0.0, dg_oh - 1.23, dg_o - 2.46, dg_ooh - 3.69, 4.92 - 4.92]
    # Levels at U = U_L
    g_ul = [0.0, dg_oh - u_l, dg_o - 2 * u_l, dg_ooh - 3 * u_l, 4.92 - 4 * u_l]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    draw_stepped_profile(ax, g_u0, "#d62728", r"$U = 0.00$ V (Standard State)")
    draw_stepped_profile(ax, g_u123, "#1f77b4", r"$U = 1.23$ V (Equilibrium Potential)")
    draw_stepped_profile(ax, g_ul, "#2ca02c", rf"$U = {u_l:.2f}$ V (Onset / Limiting Potential, $\eta = {eta:.2f}$ V)")

    x_labels = [
        r"$* + 2\mathrm{H}_2\mathrm{O}$",
        r"$*\mathrm{OH} + \mathrm{H}_2\mathrm{O} + (\mathrm{H}^+ + e^-)$",
        r"$*\mathrm{O} + \mathrm{H}_2\mathrm{O} + 2(\mathrm{H}^+ + e^-)$",
        r"$*\mathrm{OOH} + 3(\mathrm{H}^+ + e^-)$",
        r"$* + \mathrm{O}_2 + 4(\mathrm{H}^+ + e^-)$"
    ]
    ax.set_xticks(range(5))
    ax.set_xticklabels(x_labels, rotation=12, ha="right")
    ax.set_ylabel(r"Gibbs Free Energy $\Delta G$ (eV)")
    ax.set_title(rf"OER Free Energy Reaction Coordinate: {row['primary_metal'] if 'primary_metal' in row else row['metal']}-MOF ({q_id})", pad=20)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    # Inset Atomistic Visualizations
    inset_configs = [
        (0, PRISTINE_DIR / f"{q_id}.cif", "*"),
        (1, STRUCTURES_DIR / f"{q_id}_oer_OH.cif", "*OH"),
        (2, STRUCTURES_DIR / f"{q_id}_oer_O.cif", "*O"),
        (3, STRUCTURES_DIR / f"{q_id}_oer_OOH.cif", "*OOH"),
    ]

    # Map normalized coordinates for insets
    y_min, y_max = ax.get_ylim()
    inset_positions = [
        [0.02, 0.65, 0.20, 0.25],
        [0.22, 0.65, 0.20, 0.25],
        [0.44, 0.65, 0.20, 0.25],
        [0.66, 0.65, 0.20, 0.25]
    ]

    for (step_i, path, tag), pos in zip(inset_configs, inset_positions):
        if path.exists():
            ax_ins = ax.inset_axes(pos)
            cluster = extract_active_cluster(path, site_idx=0, radius=3.2)
            plot_atoms(cluster, ax_ins, radii=0.55, rotation="15x,15y,0z")
            ax_ins.axis("off")
            ax_ins.set_title(tag, fontsize=10, weight="bold", color="#333", pad=2)

    png_path = FIG_DIR / "fig4_oer_free_energy_diagram.png"
    pdf_path = FIG_DIR / "fig4_oer_free_energy_diagram.pdf"
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    logger.info(f"Saved Figure 4 to {png_path} and {pdf_path}")


def plot_her_diagram(q_id: str = "qmof-b46c098"):
    """Figure 5: HER Free Energy Diagram at 2 potentials with atomistic insets."""
    df = pd.read_csv(CHE_SUMMARY_CSV).set_index("qmof_id")
    if q_id not in df.index:
        q_id = df.index[0]
    row = df.loc[q_id]

    dg_h = row["delta_G_H_eV"]
    eta = row["eta_HER_V"]

    # Levels at U = 0 V
    g_u0 = [0.0, dg_h, 0.0]
    # Levels at U = -eta V
    g_ueta = [0.0, dg_h - eta if dg_h > 0 else dg_h + eta, 0.0]

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    draw_stepped_profile(ax, g_u0, "#d62728", r"$U = 0.00$ V (Standard State)")
    draw_stepped_profile(ax, g_ueta, "#2ca02c", rf"$U = -{eta:.2f}$ V (HER Overpotential, $\Delta G \leq 0$)")

    x_labels = [
        r"$* + (\mathrm{H}^+ + e^-)$",
        r"$*\mathrm{H}$ (Adsorbed Intermediate)",
        r"$* + \frac{1}{2}\mathrm{H}_2(\mathrm{g})$"
    ]
    ax.set_xticks(range(3))
    ax.set_xticklabels(x_labels, rotation=10, ha="right")
    ax.set_ylabel(r"Gibbs Free Energy $\Delta G$ (eV)")
    ax.set_title(rf"HER Free Energy Reaction Coordinate: {row['metal']}-MOF ({q_id})", pad=20)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9.5)

    # Insets: pristine and *H
    inset_configs = [
        (0, PRISTINE_DIR / f"{q_id}.cif", "Pristine OMS (*)", [0.10, 0.65, 0.26, 0.28]),
        (1, STRUCTURES_DIR / f"{q_id}_her_H.cif", "Hydrogen Adduct (*H)", [0.55, 0.65, 0.26, 0.28]),
    ]

    for step_i, path, tag, pos in inset_configs:
        if path.exists():
            ax_ins = ax.inset_axes(pos)
            cluster = extract_active_cluster(path, site_idx=0, radius=3.2)
            plot_atoms(cluster, ax_ins, radii=0.55, rotation="15x,15y,0z")
            ax_ins.axis("off")
            ax_ins.set_title(tag, fontsize=10, weight="bold", color="#333", pad=2)

    png_path = FIG_DIR / "fig5_her_free_energy_diagram.png"
    pdf_path = FIG_DIR / "fig5_her_free_energy_diagram.pdf"
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    logger.info(f"Saved Figure 5 to {png_path} and {pdf_path}")


def plot_co2rr_diagram(q_id: str = "qmof-07cc468"):
    """Figure 6: CO2RR (CO path) Free Energy Diagram at 2 potentials with insets."""
    df = pd.read_csv(CHE_SUMMARY_CSV).set_index("qmof_id")
    if q_id not in df.index:
        q_id = df.index[0]
    row = df.loc[q_id]

    dg_cooh = row["delta_G_COOH_eV"]
    dg_co = row["delta_G_CO_eV"]
    eta = row["eta_CO2RR_CO_V"]
    u_l = row["u_lim_CO_V"]

    # Levels at U = 0 V
    g_u0 = [0.0, dg_cooh, dg_co, -0.22]
    # Levels at U = U_L
    g_ul = [0.0, dg_cooh + u_l, dg_co + 2 * u_l, -0.22 + 2 * u_l]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    draw_stepped_profile(ax, g_u0, "#d62728", r"$U = 0.00$ V (Standard State)")
    draw_stepped_profile(ax, g_ul, "#2ca02c", rf"$U = {u_l:.2f}$ V (Onset Potential, $\eta = {eta:.2f}$ V)")

    x_labels = [
        r"$\mathrm{CO}_2(\mathrm{g}) + 2(\mathrm{H}^+ + e^-) + *$",
        r"$*\mathrm{COOH} + (\mathrm{H}^+ + e^-)$",
        r"$*\mathrm{CO} + \mathrm{H}_2\mathrm{O}$",
        r"$\mathrm{CO}(\mathrm{g}) + \mathrm{H}_2\mathrm{O} + *$"
    ]
    ax.set_xticks(range(4))
    ax.set_xticklabels(x_labels, rotation=10, ha="right")
    ax.set_ylabel(r"Gibbs Free Energy $\Delta G$ (eV)")
    ax.set_title(rf"$\mathrm{{CO}}_2\mathrm{{RR}} \rightarrow \mathrm{{CO}}$ Free Energy Diagram: {row['metal']}-MOF ({q_id})", pad=20)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    inset_configs = [
        (0, PRISTINE_DIR / f"{q_id}.cif", "Pristine OMS (*)", [0.03, 0.65, 0.24, 0.27]),
        (1, STRUCTURES_DIR / f"{q_id}_co2rr_COOH.cif", "Carboxyl (*COOH)", [0.35, 0.65, 0.24, 0.27]),
        (2, STRUCTURES_DIR / f"{q_id}_co2rr_CO.cif", "Carbonyl (*CO)", [0.67, 0.65, 0.24, 0.27]),
    ]

    for step_i, path, tag, pos in inset_configs:
        if path.exists():
            ax_ins = ax.inset_axes(pos)
            cluster = extract_active_cluster(path, site_idx=0, radius=3.2)
            plot_atoms(cluster, ax_ins, radii=0.55, rotation="15x,15y,0z")
            ax_ins.axis("off")
            ax_ins.set_title(tag, fontsize=10, weight="bold", color="#333", pad=2)

    png_path = FIG_DIR / "fig6_co2rr_free_energy_diagram.png"
    pdf_path = FIG_DIR / "fig6_co2rr_free_energy_diagram.pdf"
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    logger.info(f"Saved Figure 6 to {png_path} and {pdf_path}")


def main():
    logger.info("=== STEP 11: Reaction Coordinate Free Energy Diagrams with Atomistic Insets ===")

    if not CHE_SUMMARY_CSV.exists():
        logger.error(f"CHE summary CSV missing: {CHE_SUMMARY_CSV}")
        sys.exit(1)

    # 1. Generate OER Diagram
    plot_oer_diagram("qmof-73ded45")
    # 2. Generate HER Diagram
    plot_her_diagram("qmof-b46c098")
    # 3. Generate CO2RR Diagram
    plot_co2rr_diagram("qmof-07cc468")

    # =========================================================================
    # OUTPUT VALIDATION BLOCK (Required by Protocol Section 9.2 & 11.1)
    # =========================================================================
    validation_passed = True
    validation_errors = []

    expected = [
        FIG_DIR / "fig4_oer_free_energy_diagram.png",
        FIG_DIR / "fig4_oer_free_energy_diagram.pdf",
        FIG_DIR / "fig5_her_free_energy_diagram.png",
        FIG_DIR / "fig5_her_free_energy_diagram.pdf",
        FIG_DIR / "fig6_co2rr_free_energy_diagram.png",
        FIG_DIR / "fig6_co2rr_free_energy_diagram.pdf",
    ]

    for fpath in expected:
        if not fpath.exists() or fpath.stat().st_size == 0:
            validation_passed = False
            validation_errors.append(f"Missing or empty diagram file: {fpath.name}")

    if validation_passed:
        logger.info("\n[VALIDATION PASSED] script_09_free_energy_diagrams.py")
        logger.info(f"Successfully generated all reaction coordinate free energy diagrams with atomistic insets.")
    else:
        logger.error("\n[VALIDATION FAILED] script_09_free_energy_diagrams.py")
        for err in validation_errors:
            logger.error(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
