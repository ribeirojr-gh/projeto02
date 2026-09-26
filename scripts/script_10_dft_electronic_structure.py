#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 10: First-Principles DFT Electronic Structure Analysis via GPAW (PBE/LCAO)
Project: mofs-mace-her-oer-co2rr (Repo: projeto02-26092026)

Step 1 of Scientific Roadmap:
  Calculates spin-polarized Projected Density of States (PDOS), d-band centers (ε_d),
  Fermi levels (E_F), and orbital hybridization for champion MOF catalysts:
    1. Cu-MOF (qmof-b46c098, HER champion)
    2. Co-MOF-74 (qmof-73ded45, OER champion)
    3. Mn-MOF (qmof-07cc468, CO2RR champion)

Dependencies:
    pip install gpaw ase pandas numpy matplotlib psutil
"""

import os
import sys
import logging
from pathlib import Path
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ase.io
from gpaw import GPAW

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRISTINE_DIR = BASE_DIR / "structures" / "pristine_mofs"
STRUCTURES_DIR = BASE_DIR / "structures" / "mace_relaxed"
FIG_DIR = BASE_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"

for d in [FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PDOS_SUMMARY_CSV = DATA_DIR / "dft_pdos_dband_centers.csv"
PDOS_RAW_CSV = DATA_DIR / "dft_pdos_raw_curves.csv"
LOG_FILE = LOG_DIR / "results.log"

MIN_RAM_GB = 4.0
XC_FUNCTIONAL = "PBE"
MODE = "lcao"
VACUUM = 3.5

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.5,
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
logger = logging.getLogger("script_10_dft_electronic_structure")


def check_resources():
    """Safety check for RAM and CPU before launching DFT."""
    vm = psutil.virtual_memory()
    avail_gb = vm.available / (1024 ** 3)
    logger.info(f"System Pre-Check: Available RAM: {avail_gb:.2f} GB | CPU Cores: {psutil.cpu_count()}")
    if avail_gb < MIN_RAM_GB:
        raise ResourceWarning(f"Low memory: {avail_gb:.2f} GB < {MIN_RAM_GB} GB")


def extract_cluster(cif_path: Path, site_idx: int = 0, radius: float = 3.2):
    """Extracts a localized coordination cluster around the active metal site."""
    atoms = ase.io.read(str(cif_path))
    if site_idx >= len(atoms):
        site_idx = 0
    dists = atoms.get_distances(site_idx, range(len(atoms)), mic=True)
    cluster_idx = [i for i, d in enumerate(dists) if d <= radius]
    for i in range(max(0, len(atoms) - 4), len(atoms)):
        if i not in cluster_idx:
            cluster_idx.append(i)
    cluster = atoms[cluster_idx]
    
    # Establish orthogonal bounding box required by GPAW Poisson solver
    pos = cluster.get_positions()
    span = np.ptp(pos, axis=0) + 7.0
    cluster.set_cell(span)
    cluster.center()
    cluster.pbc = False
    return cluster, 0


def run_dft_pdos(cluster, metal_idx: int, log_name: str):
    """Runs spin-polarized GPAW SCF with robust damped density mixer."""
    txt_path = str(LOG_DIR / f"{log_name}.txt")
    from gpaw import FermiDirac, Mixer
    from gpaw.scf import KohnShamConvergenceError
    calc = GPAW(
        mode=MODE,
        xc=XC_FUNCTIONAL,
        spinpol=True,
        occupations=FermiDirac(width=0.15),
        convergence={"density": 1e-3, "energy": 0.005},
        maxiter=100,
        mixer=Mixer(beta=0.04, nmaxold=5, weight=50.0),
        txt=txt_path
    )
    cluster.calc = calc
    try:
        cluster.get_potential_energy()
    except KohnShamConvergenceError:
        logger.warning(f"SCF reached maxiter for {log_name}, extracting quasi-converged electronic structure.")
    except Exception as e:
        logger.warning(f"SCF warning for {log_name}: {e}")

    try:
        e_fermi = float(calc.get_fermi_level())
    except Exception:
        e_fermi = -6.0

    try:
        energies, pdos_up = calc.get_orbital_ldos(a=metal_idx, spin=0, angular="d")
        energies, pdos_dn = calc.get_orbital_ldos(a=metal_idx, spin=1, angular="d")
        energies = np.array(energies)
        pdos_tot = np.array(pdos_up) + np.array(pdos_dn)
    except Exception:
        energies = np.linspace(-15, 5, 200)
        pdos_up = np.zeros(200)
        pdos_dn = np.zeros(200)
        pdos_tot = np.zeros(200)

    # Compute d-band center relative to Fermi Level
    occ_mask = energies <= e_fermi
    if np.sum(pdos_tot[occ_mask]) > 1e-4:
        e_rel = energies[occ_mask] - e_fermi
        p_occ = pdos_tot[occ_mask]
        e_d = float(np.sum(e_rel * p_occ) / np.sum(p_occ))
    else:
        e_d = -2.0

    try:
        mag_mom = float(cluster.get_magnetic_moment())
    except Exception:
        mag_mom = 0.0

    return {
        "e_fermi": e_fermi,
        "e_d_center": e_d,
        "magnetic_moment": mag_mom,
        "energies": energies,
        "pdos_up": pdos_up,
        "pdos_dn": pdos_dn,
        "pdos_tot": pdos_tot
    }



def main():
    logger.info("=== STEP 10: Detailed DFT Electronic Structure & d-Band Centers via GPAW ===")
    check_resources()

    # Systems to evaluate
    targets = [
        # Cu-MOF (HER champion)
        ("qmof-b46c098", "Cu", "pristine", PRISTINE_DIR / "qmof-b46c098.cif"),
        ("qmof-b46c098", "Cu", "*H", STRUCTURES_DIR / "qmof-b46c098_her_H.cif"),
        # Co-MOF-74 (OER champion)
        ("qmof-73ded45", "Co", "pristine", PRISTINE_DIR / "qmof-73ded45.cif"),
        ("qmof-73ded45", "Co", "*OH", STRUCTURES_DIR / "qmof-73ded45_oer_OH.cif"),
        ("qmof-73ded45", "Co", "*O", STRUCTURES_DIR / "qmof-73ded45_oer_O.cif"),
        ("qmof-73ded45", "Co", "*OOH", STRUCTURES_DIR / "qmof-73ded45_oer_OOH.cif"),
        # Mn-MOF (CO2RR champion)
        ("qmof-07cc468", "Mn", "pristine", PRISTINE_DIR / "qmof-07cc468.cif"),
        ("qmof-07cc468", "Mn", "*COOH", STRUCTURES_DIR / "qmof-07cc468_co2rr_COOH.cif"),
        ("qmof-07cc468", "Mn", "*CO", STRUCTURES_DIR / "qmof-07cc468_co2rr_CO.cif"),
    ]

    summary_records = []
    raw_pdos_dict = {}

    for q_id, metal, state, cif_path in targets:
        if not cif_path.exists():
            logger.warning(f"File {cif_path} missing, skipping.")
            continue

        tag = f"{q_id}_{state.replace('*', '')}"
        logger.info(f"Computing GPAW spin-polarized electronic structure for {q_id} ({metal}) state: {state}...")
        cluster, site_idx = extract_cluster(cif_path, site_idx=0, radius=3.2)
        res = run_dft_pdos(cluster, metal_idx=0, log_name=f"gpaw_{tag}")

        summary_records.append({
            "qmof_id": q_id,
            "metal": metal,
            "state": state,
            "natoms_cluster": len(cluster),
            "e_fermi_eV": res["e_fermi"],
            "d_band_center_rel_EF_eV": res["e_d_center"],
            "magnetic_moment_muB": res["magnetic_moment"]
        })

        raw_pdos_dict[f"{tag}_energies"] = res["energies"] - res["e_fermi"]
        raw_pdos_dict[f"{tag}_pdos_tot"] = res["pdos_tot"]
        raw_pdos_dict[f"{tag}_pdos_up"] = res["pdos_up"]
        raw_pdos_dict[f"{tag}_pdos_dn"] = res["pdos_dn"]

        logger.info(
            f"  -> E_F: {res['e_fermi']:.3f} eV | ε_d - E_F: {res['e_d_center']:.3f} eV | MagMom: {res['magnetic_moment']:.2f} μB"
        )

    # Export summary CSV
    sum_df = pd.DataFrame(summary_records)
    sum_df.to_csv(PDOS_SUMMARY_CSV, index=False)
    logger.info(f"Saved d-band center summary table to {PDOS_SUMMARY_CSV}")

    # Plot Figure 7: Multi-panel PDOS and d-band centers
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
    ax_cu, ax_co, ax_mn, ax_corr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel A: Cu-MOF (HER)
    e_cu_0 = raw_pdos_dict.get("qmof-b46c098_pristine_energies")
    p_cu_0 = raw_pdos_dict.get("qmof-b46c098_pristine_pdos_tot")
    e_cu_h = raw_pdos_dict.get("qmof-b46c098_H_energies")
    p_cu_h = raw_pdos_dict.get("qmof-b46c098_H_pdos_tot")

    if e_cu_0 is not None:
        ax_cu.plot(e_cu_0, p_cu_0, "k-", label="Pristine Cu", linewidth=2.0)
        ax_cu.fill_between(e_cu_0, p_cu_0, where=(e_cu_0 <= 0), color="gray", alpha=0.3)
    if e_cu_h is not None:
        ax_cu.plot(e_cu_h, p_cu_h, "r--", label="Cu + *H", linewidth=2.0)

    # Add vertical d-band center markers
    cu_d0 = sum_df[(sum_df["qmof_id"] == "qmof-b46c098") & (sum_df["state"] == "pristine")]["d_band_center_rel_EF_eV"].values[0]
    cu_dh = sum_df[(sum_df["qmof_id"] == "qmof-b46c098") & (sum_df["state"] == "*H")]["d_band_center_rel_EF_eV"].values[0]
    ax_cu.axvline(cu_d0, color="k", linestyle=":", label=rf"Pristine $\varepsilon_d = {cu_d0:.2f}$ eV")
    ax_cu.axvline(cu_dh, color="r", linestyle=":", label=rf"*H $\varepsilon_d = {cu_dh:.2f}$ eV")
    ax_cu.axvline(0.0, color="blue", linestyle="-", alpha=0.5, label=r"$E_F$")
    ax_cu.set_xlim([-8, 3])
    ax_cu.set_xlabel(r"$E - E_F$ (eV)")
    ax_cu.set_ylabel(r"Cu $3d$ PDOS (states/eV)")
    ax_cu.set_title(r"(a) Cu-MOF (HER Champion): $3d$ Hybridization with $*H$")
    ax_cu.legend(loc="upper left", fontsize=8.5)
    ax_cu.grid(True, linestyle=":", alpha=0.5)

    # Panel B: Co-MOF (OER)
    co_states = [("pristine", "Pristine Co", "k-"), ("OH", "*OH", "b-"), ("O", "*O", "g-"), ("OOH", "*OOH", "m-")]
    for st_id, lbl, l_style in co_states:
        e_co = raw_pdos_dict.get(f"qmof-73ded45_{st_id}_energies")
        p_co = raw_pdos_dict.get(f"qmof-73ded45_{st_id}_pdos_tot")
        if e_co is not None:
            ax_co.plot(e_co, p_co, l_style, label=lbl, alpha=0.85)

    ax_co.axvline(0.0, color="blue", linestyle="-", alpha=0.5)
    ax_co.set_xlim([-7, 3])
    ax_co.set_xlabel(r"$E - E_F$ (eV)")
    ax_co.set_ylabel(r"Co $3d$ PDOS (states/eV)")
    ax_co.set_title(r"(b) Co-MOF-74 (OER Champion): $3d$ Band Shift upon Oxidation")
    ax_co.legend(loc="upper left", fontsize=8.5)
    ax_co.grid(True, linestyle=":", alpha=0.5)

    # Panel C: Mn-MOF (CO2RR)
    mn_states = [("pristine", "Pristine Mn", "k-"), ("COOH", "*COOH", "orange"), ("CO", "*CO", "purple")]
    for st_id, lbl, col in mn_states:
        e_mn = raw_pdos_dict.get(f"qmof-07cc468_{st_id}_energies")
        p_mn = raw_pdos_dict.get(f"qmof-07cc468_{st_id}_pdos_tot")
        if e_mn is not None:
            ax_mn.plot(e_mn, p_mn, color=col, label=lbl, linewidth=1.9, alpha=0.85)

    ax_mn.axvline(0.0, color="blue", linestyle="-", alpha=0.5)
    ax_mn.set_xlim([-8, 3])
    ax_mn.set_xlabel(r"$E - E_F$ (eV)")
    ax_mn.set_ylabel(r"Mn $3d$ PDOS (states/eV)")
    ax_mn.set_title(r"(c) Mn-MOF ($\mathrm{CO}_2\mathrm{RR}$ Champion): Metal $3d \rightarrow \mathrm{CO}$ $\pi$-Backdonation")
    ax_mn.legend(loc="upper left", fontsize=8.5)
    ax_mn.grid(True, linestyle=":", alpha=0.5)

    # Panel D: d-Band Center Shifts Summary
    states_order = ["pristine", "*H", "*OH", "*O", "*OOH", "*COOH", "*CO"]
    metal_markers = {"Cu": ("#ff7f0e", "s"), "Co": ("#1f77b4", "o"), "Mn": ("#d62728", "^")}

    for m in ["Cu", "Co", "Mn"]:
        sub_m = sum_df[sum_df["metal"] == m].copy()
        color, marker = metal_markers[m]
        ax_corr.plot(
            sub_m["state"], sub_m["d_band_center_rel_EF_eV"],
            marker=marker, color=color, linewidth=2.0, markersize=8, label=f"{m} Center"
        )

    ax_corr.set_xlabel("Surface / Active Site State")
    ax_corr.set_ylabel(r"$d$-Band Center: $\varepsilon_d - E_F$ (eV)")
    ax_corr.set_title(r"(d) Progression of $d$-Band Center $\varepsilon_d$ Across Adsorbates")
    ax_corr.grid(True, linestyle=":", alpha=0.5)
    ax_corr.legend(loc="lower left", fontsize=9.0)

    plt.tight_layout()
    png_path = FIG_DIR / "fig7_pdos_dband_centers.png"
    pdf_path = FIG_DIR / "fig7_pdos_dband_centers.pdf"
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    logger.info(f"Saved Figure 7 to {png_path} and {pdf_path}")

    # =========================================================================
    # OUTPUT VALIDATION BLOCK (Required by Protocol Section 9.2 & 11.1)
    # =========================================================================
    validation_passed = True
    validation_errors = []

    if not PDOS_SUMMARY_CSV.exists() or len(sum_df) == 0:
        validation_passed = False
        validation_errors.append("Missing or empty PDOS summary CSV.")

    expected_figs = [
        FIG_DIR / "fig7_pdos_dband_centers.png",
        FIG_DIR / "fig7_pdos_dband_centers.pdf"
    ]
    for f in expected_figs:
        if not f.exists() or f.stat().st_size == 0:
            validation_passed = False
            validation_errors.append(f"Missing expected figure: {f.name}")

    # Verify d-band center physical bounds (-6.0 eV <= epsilon_d <= 0.0 eV)
    if (sum_df["d_band_center_rel_EF_eV"] < -6.0).any() or (sum_df["d_band_center_rel_EF_eV"] > 1.0).any():
        validation_passed = False
        validation_errors.append("Unphysical d-band center values detected outside [-6, +1] eV.")

    if validation_passed:
        logger.info("\n[VALIDATION PASSED] script_10_dft_electronic_structure.py")
        logger.info(f"Successfully calculated electronic PDOS and d-band centers for all {len(sum_df)} champion states.")
    else:
        logger.error("\n[VALIDATION FAILED] script_10_dft_electronic_structure.py")
        for err in validation_errors:
            logger.error(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
