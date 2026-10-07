#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 12: High-Throughput Electrocatalytic Scale-Up Across Synthesized MOFs
Project: mofs-mace-her-oer-co2rr (Repo: projeto02-26092026)

Step 3 of Approved Scientific Roadmap:
  Expands electrocatalytic mechanism discovery to a broad candidate cohort
  spanning all 9 target transition metals (Co, Cu, Fe, Mn, Mo, Ni, Ru, Zn, Zr)
  screened from the 1,514 synthesized MOFs in the Materials Project / QMOF database.

Methodology:
  - Ingestion of diverse synthesized MOFs (PLD >= 3.0 Å, Natoms <= 150)
  - Automatic Open Metal Site (OMS) and outward coordination vector extraction
  - Construction of reaction intermediates:
      * HER: *H
      * OER: *OH, *O, *OOH
      * CO2RR: *COOH, *CO
  - GPU-accelerated MACE-MP-0 ground-state relaxation (fmax < 0.05 eV/Å)
  - CHE thermodynamic evaluation of overpotentials and selectivity
  - Multi-descriptor correlation analysis (PLD, Bandgap, d-electron count)
  - Generation of 4-panel publication Figure 9 (PNG & PDF) and SI LaTeX Table S3

Dependencies:
  mace-torch ase pandas numpy matplotlib torch pymatgen
"""

import os
import sys
import io
import zipfile
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ase.io
from ase import Atoms, Atom
from ase.optimize import BFGS
from mace.calculators import mace_mp

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRISTINE_DIR = BASE_DIR / "structures" / "scaleup_pristine"
INTERMEDIATE_DIR = BASE_DIR / "structures" / "scaleup_intermediates"
FIG_DIR = BASE_DIR / "figures"
SI_DIR = BASE_DIR / "SI" / "tables"
LOG_DIR = BASE_DIR / "logs"

for d in [PRISTINE_DIR, INTERMEDIATE_DIR, FIG_DIR, SI_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ZIP_PATH = DATA_DIR / "qmof_database.zip"
FILTERED_CSV = DATA_DIR / "metadata_qmof_filtered.csv"
OUTPUT_CSV = DATA_DIR / "high_throughput_scaleup_results.csv"
TEX_OUTPUT = SI_DIR / "table3_high_throughput_summary.tex"
LOG_FILE = LOG_DIR / "results.log"

TARGET_METALS = ["Co", "Cu", "Fe", "Mn", "Mo", "Ni", "Ru", "Zn", "Zr"]
MAX_CANDIDATES_PER_METAL = 4
FMAX_SCALEUP = 0.03      # aligned with script_04/manuscript criterion (was 0.05)
MAX_STEPS = 300           # raised from 25 (see script_04 for the same audit finding)
MODEL_SIZE = "medium"     # matches manuscript's stated Methods (was "small")
DTYPE = "float64"
DEVICE = "cuda"

# Standard thermochemical corrections from Step 7 (PHVA localized vibrations)
THERMO_CORR = {
    "*H": {"zpe": 0.27, "ts": 0.02, "dG_corr": 0.25},
    "*OH": {"zpe": 0.38, "ts": 0.05, "dG_corr": 0.33},
    "*O": {"zpe": 0.09, "ts": 0.04, "dG_corr": 0.05},
    "*OOH": {"zpe": 0.44, "ts": 0.07, "dG_corr": 0.37},
    "*COOH": {"zpe": 0.61, "ts": 0.12, "dG_corr": 0.49},
    "*CO": {"zpe": 0.18, "ts": 0.08, "dG_corr": 0.10}
}

# Gas Phase MACE-MP-0 Ground State Reference Energies
GAS_REFS = {
    "H2": -6.74311,
    "H2O": -14.05150,
    "CO2": -23.10984,
    "CO": -14.77452
}

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{amsmath}\usepackage{amssymb}",
    "font.size": 10,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.8,
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
logger = logging.getLogger("script_12_scaleup")


def select_cohort(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Selects a balanced, high-diversity cohort of synthesized MOFs."""
    cond = (
        (meta_df["info.synthesized"] == True) &
        (meta_df["info.pld"] >= 3.0) &
        (meta_df["info.natoms"] <= 140)
    )
    filtered = meta_df[cond].copy()
    
    selected_rows = []
    for m in TARGET_METALS:
        sub_m = filtered[filtered["primary_metal"] == m]
        if len(sub_m) > 0:
            # Sort by pore size and moderate unit cell size
            sub_m = sub_m.sort_values(by=["info.pld", "info.natoms"], ascending=[False, True])
            selected_rows.append(sub_m.head(MAX_CANDIDATES_PER_METAL))
            
    cohort = pd.concat(selected_rows).reset_index(drop=True)
    logger.info(f"Selected {len(cohort)} diverse synthesized MOFs across {len(cohort['primary_metal'].unique())} metal families.")
    return cohort


def extract_cif_from_zip(nested_zf: zipfile.ZipFile, qmof_id: str, out_path: Path) -> bool:
    """Extracts a pristine CIF from the QMOF archive."""
    target_name = f"{qmof_id}.cif"
    for name in nested_zf.namelist():
        if name.endswith(target_name):
            data = nested_zf.read(name)
            with open(out_path, "wb") as f:
                f.write(data)
            return True
    return False


def build_adsorbate(atoms: Atoms, metal_idx: int, ads_type: str, bond_dist: float, u_open: np.ndarray) -> Atoms:
    """Attaches an adsorbate along the outward pore normal u_open."""
    ads_atoms = atoms.copy()
    m_pos = atoms.positions[metal_idx]
    u = u_open / np.linalg.norm(u_open)

    if ads_type == "*H":
        pos_h = m_pos + bond_dist * u
        ads_atoms.append(Atom("H", position=pos_h))

    elif ads_type == "*OH":
        pos_o = m_pos + bond_dist * u
        pos_h = pos_o + 0.96 * (u + np.array([0.4, 0.4, 0.0])) / np.linalg.norm(u + np.array([0.4, 0.4, 0.0]))
        ads_atoms.append(Atom("O", position=pos_o))
        ads_atoms.append(Atom("H", position=pos_h))

    elif ads_type == "*O":
        pos_o = m_pos + bond_dist * u
        ads_atoms.append(Atom("O", position=pos_o))

    elif ads_type == "*OOH":
        pos_o1 = m_pos + bond_dist * u
        perp = np.cross(u, [0, 0, 1]) if abs(u[2]) < 0.9 else np.cross(u, [0, 1, 0])
        perp /= np.linalg.norm(perp)
        pos_o2 = pos_o1 + 1.35 * (0.7 * u + 0.7 * perp)
        pos_h = pos_o2 + 0.97 * (0.3 * u - 0.95 * perp)
        ads_atoms.append(Atom("O", position=pos_o1))
        ads_atoms.append(Atom("O", position=pos_o2))
        ads_atoms.append(Atom("H", position=pos_h))

    elif ads_type == "*COOH":
        pos_c = m_pos + bond_dist * u
        perp = np.cross(u, [0, 0, 1]) if abs(u[2]) < 0.9 else np.cross(u, [0, 1, 0])
        perp /= np.linalg.norm(perp)
        pos_o1 = pos_c + 1.22 * (0.8 * u + 0.6 * perp)
        pos_o2 = pos_c + 1.35 * (0.8 * u - 0.6 * perp)
        pos_h = pos_o2 + 0.96 * perp
        ads_atoms.append(Atom("C", position=pos_c))
        ads_atoms.append(Atom("O", position=pos_o1))
        ads_atoms.append(Atom("O", position=pos_o2))
        ads_atoms.append(Atom("H", position=pos_h))

    elif ads_type == "*CO":
        pos_c = m_pos + bond_dist * u
        pos_o = pos_c + 1.15 * u
        ads_atoms.append(Atom("C", position=pos_c))
        ads_atoms.append(Atom("O", position=pos_o))

    return ads_atoms


def main():
    logger.info("=== STEP 12: High-Throughput Electrocatalytic Scale-Up Across Synthesized MOFs ===")
    
    if not FILTERED_CSV.exists():
        logger.error("Required dataset file missing: %s", FILTERED_CSV)
        sys.exit(1)
    if not ZIP_PATH.exists():
        logger.warning(
            f"{ZIP_PATH} not present (it is .gitignore'd as a >100MB raw archive and "
            "was never committed). Proceeding without it: all 32 pristine CIFs already "
            "exist locally in structures/scaleup_pristine/ from the original run, so the "
            "zip is only needed as a fallback for any structure missing locally."
        )

    if OUTPUT_CSV.exists():
        logger.info(f"Loading existing high-throughput screening data from {OUTPUT_CSV}...")
        res_df = pd.read_csv(OUTPUT_CSV)
    else:
        meta_df = pd.read_csv(FILTERED_CSV)
        cohort = select_cohort(meta_df)

        # Initialize MACE-MP-0 on GPU
        logger.info(f"Initializing MACE-MP-0 ({MODEL_SIZE}) on {DEVICE}...")
        calc = mace_mp(model=MODEL_SIZE, device=DEVICE, default_dtype=DTYPE)

        nested_zf = None
        if ZIP_PATH.exists():
            zf = zipfile.ZipFile(ZIP_PATH, "r")
            nested_bytes = zf.read("qmof_database/relaxed_structures.zip")
            nested_zf = zipfile.ZipFile(io.BytesIO(nested_bytes))
        results = []

        for idx, row in cohort.iterrows():
            q_id = row["qmof_id"]
            metal = row["primary_metal"]
            pld = row["info.pld"]
            lcd = row["info.lcd"]
            bandgap = row.get("outputs.pbe.bandgap", 1.5)
            
            pristine_cif = PRISTINE_DIR / f"{q_id}.cif"
            if not pristine_cif.exists():
                if nested_zf is None or not extract_cif_from_zip(nested_zf, q_id, pristine_cif):
                    logger.warning(f"Could not find or extract CIF for {q_id} (no local file, no zip available), skipping.")
                    continue

            try:
                atoms = ase.io.read(str(pristine_cif))
            except Exception as e:
                logger.warning(f"Failed to read CIF for {q_id}: {e}")
                continue

            # Find transition metal active site
            metal_indices = [i for i, at in enumerate(atoms) if at.symbol == metal]
            if not metal_indices:
                continue
            m_idx = metal_indices[0]

            # Calculate coordination and pore normal
            dists = atoms.get_distances(m_idx, range(len(atoms)), mic=True)
            coord_indices = [i for i, d in enumerate(dists) if 0.1 < d < 2.5]
            coord_num = len(coord_indices)
            
            if coord_indices:
                vecs = atoms.get_distances(m_idx, coord_indices, mic=True, vector=True)
                u_open = -np.mean(vecs, axis=0)
                if np.linalg.norm(u_open) < 0.1:
                    u_open = np.array([0.0, 0.0, 1.0])
                else:
                    u_open /= np.linalg.norm(u_open)
            else:
                u_open = np.array([0.0, 0.0, 1.0])

            logger.info(f"[{idx+1}/{len(cohort)}] Screening {q_id} ({metal}, PLD={pld:.2f} Å, Coord={coord_num})...")

            # 1. Clean MOF ground state
            atoms.calc = calc
            dyn = BFGS(atoms, logfile=None)
            dyn.run(fmax=FMAX_SCALEUP, steps=MAX_STEPS)
            e_clean = float(atoms.get_potential_energy())

            # Intermediates to screen
            inter_configs = [
                ("*H", 1.55),
                ("*OH", 1.95),
                ("*O", 1.80),
                ("*OOH", 1.95),
                ("*COOH", 1.95),
                ("*CO", 1.90)
            ]

            energies = {}
            any_congested = False
            for ads_type, b_dist in inter_configs:
                struct = build_adsorbate(atoms, m_idx, ads_type, b_dist, u_open)
                struct.calc = calc
                dyn = BFGS(struct, logfile=None)
                dyn.run(fmax=FMAX_SCALEUP, steps=MAX_STEPS)
                e_tot = float(struct.get_potential_energy())
                fmax_val = float(np.max(np.linalg.norm(struct.get_forces(), axis=1)))

                # Check for steric clash / non-convergence. A prior version of
                # this script substituted a fixed "+10 eV boundary penalty"
                # here, which silently propagated into eta_OER/eta_CO2RR as
                # double-digit-volt values reaching the published figure. We
                # now flag the structure as congested (NaN its energy) instead
                # of injecting a fabricated number; congested rows are dropped
                # before plotting, matching script_07's convention.
                if fmax_val > 5.0 or abs(e_tot - e_clean) > 6.0:
                    logger.warning(f"Steric clash/congestion in {q_id} {ads_type} (fmax={fmax_val:.1f} eV/Å). Flagging, not penalizing.")
                    e_tot = np.nan
                    any_congested = True

                energies[ads_type] = e_tot

            # Calculate reaction free energies (CHE, U=0 V, pH=0)
            dE_H = energies["*H"] - e_clean - 0.5 * GAS_REFS["H2"]
            dG_H = dE_H + THERMO_CORR["*H"]["dG_corr"]
            eta_HER = abs(dG_H)

            # OER:
            dE_OH = energies["*OH"] - e_clean - (GAS_REFS["H2O"] - 0.5 * GAS_REFS["H2"])
            dG_OH = dE_OH + THERMO_CORR["*OH"]["dG_corr"]

            dE_O = energies["*O"] - e_clean - (GAS_REFS["H2O"] - GAS_REFS["H2"])
            dG_O = dE_O + THERMO_CORR["*O"]["dG_corr"]

            dE_OOH = energies["*OOH"] - e_clean - (2 * GAS_REFS["H2O"] - 1.5 * GAS_REFS["H2"])
            dG_OOH = dE_OOH + THERMO_CORR["*OOH"]["dG_corr"]

            dg1 = dG_OH
            dg2 = dG_O - dG_OH
            dg3 = dG_OOH - dG_O
            dg4 = 4.92 - dG_OOH
            pds_val = max([dg1, dg2, dg3, dg4])
            eta_OER = max(0.0, pds_val - 1.23)

            # CO2RR:
            dE_COOH = energies["*COOH"] - e_clean - (GAS_REFS["CO2"] + 0.5 * GAS_REFS["H2"])
            dG_COOH = dE_COOH + THERMO_CORR["*COOH"]["dG_corr"]

            dE_CO = energies["*CO"] - e_clean - (GAS_REFS["CO2"] + GAS_REFS["H2"] - GAS_REFS["H2O"])
            dG_CO = dE_CO + THERMO_CORR["*CO"]["dG_corr"]

            dg_co2_1 = dG_COOH
            dg_co2_2 = dG_CO - dG_COOH
            eta_CO2RR = max(0.0, max(dg_co2_1, dg_co2_2) - (-0.11))

            delta_G_sel = dG_COOH - dG_H

            results.append({
                "qmof_id": q_id,
                "metal": metal,
                "pld_A": pld,
                "lcd_A": lcd,
                "bandgap_eV": bandgap,
                "coord_number": coord_num,
                "natoms": len(atoms),
                "dG_H_eV": dG_H,
                "eta_HER_V": eta_HER,
                "dG_OH_eV": dG_OH,
                "dG_O_eV": dG_O,
                "dG_OOH_eV": dG_OOH,
                "eta_OER_V": eta_OER,
                "dG_COOH_eV": dG_COOH,
                "dG_CO_eV": dG_CO,
                "eta_CO2RR_V": eta_CO2RR,
                "delta_G_sel_eV": delta_G_sel,
                "prefers_CO2RR": (delta_G_sel < 0.0),
                "steric_congested": any_congested,
            })

        res_df = pd.DataFrame(results)
        res_df.to_csv(OUTPUT_CSV, index=False)
        if ZIP_PATH.exists():
            zf.close()
        logger.info(f"Saved high-throughput screening data ({len(res_df)} systems) to {OUTPUT_CSV}")

    # =========================================================================
    # PLOTTING FIGURE 9: 4-PANEL PUBLICATION SCALE-UP SUMMARY
    # =========================================================================
    if "steric_congested" in res_df.columns:
        n_before = len(res_df)
        res_df = res_df[res_df["steric_congested"] == False].copy()
        n_dropped = n_before - len(res_df)
        if n_dropped > 0:
            logger.warning(
                f"Excluding {n_dropped}/{n_before} sterically congested rows before plotting "
                f"Figure 9 (same convention as script_07 for Figures 1-3). This filter was "
                f"absent in the prior version of this script, which let eta_OER_V values of "
                f"10-24 V reach the published panel (d)."
            )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    ax_her = axes[0, 0]
    ax_oer = axes[0, 1]
    ax_sel = axes[1, 0]
    ax_desc = axes[1, 1]

    metal_palette = {
        "Co": "#1f77b4", "Cu": "#ff7f0e", "Fe": "#2ca02c", "Mn": "#d62728",
        "Mo": "#9467bd", "Ni": "#8c564b", "Ru": "#e377c2", "Zn": "#7f7f7f", "Zr": "#bcbd22"
    }

    # --- Panel A: HER Sabatier Volcano Curve ---
    x_volc = np.linspace(-1.5, 1.5, 200)
    ax_her.plot(x_volc, np.abs(x_volc), "k--", label=r"Sabatier Volcano Limit: $\eta = |\Delta G_{*\mathrm{H}}|$")
    for m in sorted(res_df["metal"].unique()):
        sub = res_df[res_df["metal"] == m]
        ax_her.scatter(sub["dG_H_eV"], sub["eta_HER_V"], color=metal_palette.get(m, "gray"),
                       s=75, edgecolors="k", alpha=0.9, label=m)

    ax_her.set_xlim([-1.3, 1.3])
    ax_her.set_ylim([0, 1.65])
    ax_her.set_xlabel(r"$\Delta G_{*\mathrm{H}}$ (eV)")
    ax_her.set_ylabel(r"HER Overpotential $\eta^{\mathrm{HER}}$ (V)")
    ax_her.set_title(r"(a) HER Sabatier Volcano Across Transition Metal MOF Cohort")
    ax_her.legend(ncol=4, fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False)
    ax_her.grid(True, linestyle=":", alpha=0.5, linewidth=0.5)

    # --- Panel B: OER Scaling Relation ---
    x_oer = np.linspace(-0.2, 2.8, 100)
    valid_oer = res_df[(res_df["dG_OH_eV"] > -2.0) & (res_df["dG_OOH_eV"] < 8.0)]
    if len(valid_oer) > 5:
        p = np.polyfit(valid_oer["dG_OH_eV"], valid_oer["dG_OOH_eV"], 1)
        ax_oer.plot(x_oer, p[0] * x_oer + p[1], "r-", linewidth=2.0,
                    label=rf"Cohort Scaling: $\Delta G_{{*OOH}} = {p[0]:.2f}\Delta G_{{*OH}} + {p[1]:.2f}$ eV")
    ax_oer.plot(x_oer, 0.95 * x_oer + 3.20, "k--", label=r"Universal Oxide Scaling: $\Delta G + 3.20$ eV")
    ax_oer.plot(x_oer, x_oer + 2.46, "g:", label=r"Ideal Non-Overpotential Limit ($\Delta G + 2.46$ eV)")

    for m in sorted(res_df["metal"].unique()):
        sub = valid_oer[valid_oer["metal"] == m]
        ax_oer.scatter(sub["dG_OH_eV"], sub["dG_OOH_eV"], color=metal_palette.get(m, "gray"),
                       s=75, edgecolors="k", alpha=0.9)

    ax_oer.set_xlim([0.0, 2.6])
    ax_oer.set_ylim([3.0, 6.8])
    ax_oer.set_xlabel(r"$\Delta G_{*\mathrm{OH}}$ (eV)")
    ax_oer.set_ylabel(r"$\Delta G_{*\mathrm{OOH}}$ (eV)")
    ax_oer.set_title(r"(b) OER Universal Scaling Relation Breakdown Across MOFs")
    ax_oer.legend(loc="upper left", frameon=False, fontsize=8.0)
    ax_oer.grid(True, linestyle=":", alpha=0.5, linewidth=0.5)

    # --- Panel C: 2D Selectivity Map (CO2RR vs HER) ---
    diag = np.linspace(-1.5, 2.0, 100)
    ax_sel.plot(diag, diag, "k-", linewidth=1.5, label=r"Equi-selectivity boundary ($\Delta G_{*\mathrm{COOH}} = \Delta G_{*\mathrm{H}}$)")
    ax_sel.fill_between(diag, diag, 3.0, color="#d95f02", alpha=0.15, label=r"Favors HER (Parasitic)")
    ax_sel.fill_between(diag, -2.0, diag, color="#1b9e77", alpha=0.15, label=r"Favors $\mathrm{CO}_2\mathrm{RR}$")

    for m in sorted(res_df["metal"].unique()):
        sub = res_df[res_df["metal"] == m]
        ax_sel.scatter(sub["dG_H_eV"], sub["dG_COOH_eV"], color=metal_palette.get(m, "gray"),
                       s=75, edgecolors="k", alpha=0.9, label=m)

    ax_sel.set_xlim([-1.2, 1.5])
    ax_sel.set_ylim([-1.2, 2.6])
    ax_sel.set_xlabel(r"$\Delta G_{*\mathrm{H}}$ (eV)")
    ax_sel.set_ylabel(r"$\Delta G_{*\mathrm{COOH}}$ (eV)")
    ax_sel.set_title(r"(c) $\mathrm{CO}_2\mathrm{RR}$ vs HER Selectivity Map")
    ax_sel.legend(ncol=2, fontsize=8.0, loc="lower right", frameon=False)
    ax_sel.grid(True, linestyle=":", alpha=0.5, linewidth=0.5)

    # --- Panel D: Descriptor Correlation (PLD & Bandgap vs Overpotential) ---
    sc = ax_desc.scatter(res_df["pld_A"], res_df["eta_OER_V"], c=res_df["bandgap_eV"],
                         cmap="viridis", s=90, edgecolors="k", alpha=0.9)
    cbar = plt.colorbar(sc, ax=ax_desc)
    cbar.set_label("DFT Bandgap (eV)")

    ax_desc.set_xlabel(r"Pore Limiting Diameter: PLD ($\mathrm{\AA}$)")
    ax_desc.set_ylabel(r"OER Overpotential $\eta^{\mathrm{OER}}$ (V)")
    ax_desc.set_title(r"(d) Structural-Electronic Activity Correlation")
    ax_desc.grid(True, linestyle=":", alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    png_path = FIG_DIR / "fig9_high_throughput_scaling_distributions.png"
    pdf_path = FIG_DIR / "fig9_high_throughput_scaling_distributions.pdf"
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    logger.info(f"Saved Figure 9 to {png_path} and {pdf_path}")

    # =========================================================================
    # GENERATE LATEX SI TABLE 3
    # =========================================================================
    with open(TEX_OUTPUT, "w", encoding="utf-8") as f:
        f.write("% Table S3: High-Throughput Electrocatalytic Screening Summary across Synthesized MOFs\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\scriptsize\n")
        f.write("\\caption{High-throughput screening results for the diverse cohort of synthesized MOFs spanning 9 transition metal families.}\n")
        f.write("\\label{tab:high_throughput_summary}\n")
        f.write("\\begin{tabular}{llccccccccc}\n")
        f.write("\\hline\\hline\n")
        f.write("QMOF ID & Metal & PLD (\\AA) & Bandgap (eV) & $\\Delta G_{*\\mathrm{H}}$ (eV) & $\\eta^{\\mathrm{HER}}$ (V) & $\\Delta G_{*\\mathrm{OH}}$ (eV) & $\\Delta G_{*\\mathrm{OOH}}$ (eV) & $\\eta^{\\mathrm{OER}}$ (V) & $\\eta^{\\mathrm{CO2RR}}$ (V) & $\\Delta G_{\\mathrm{sel}}$ (eV) \\\\\n")
        f.write("\\hline\n")
        for _, r in res_df.head(30).iterrows():
            f.write(f"{r['qmof_id']} & {r['metal']} & {r['pld_A']:.2f} & {r['bandgap_eV']:.2f} & {r['dG_H_eV']:.2f} & {r['eta_HER_V']:.2f} & {r['dG_OH_eV']:.2f} & {r['dG_OOH_eV']:.2f} & {r['eta_OER_V']:.2f} & {r['eta_CO2RR_V']:.2f} & {r['delta_G_sel_eV']:.2f} \\\\\n")
        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    logger.info(f"Saved LaTeX SI Table S3 to {TEX_OUTPUT}")

    # Output validation block
    if OUTPUT_CSV.exists() and png_path.exists() and pdf_path.exists() and TEX_OUTPUT.exists():
        logger.info("\n[VALIDATION PASSED] script_12_high_throughput_scaleup.py completed successfully.")
    else:
        logger.error("\n[VALIDATION FAILED] Missing outputs in script_12.")
        sys.exit(1)


if __name__ == "__main__":
    main()
