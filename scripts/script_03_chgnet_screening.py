#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 03: Tier-1 CHGNet High-Throughput Screening & Pre-Relaxation
Project: mofs-mace-her-oer-co2rr

Purpose:
  Rapidly relaxes pristine MOFs and intermediate complexes (HER, OER, CO2RR)
  using CHGNet on CUDA GPU. Verifies framework stability and filters anomalies.

Dependencies:
    pip install chgnet ase pymatgen pandas torch
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import ase.io
from ase.optimize import BFGS
from chgnet.model.dynamics import CHGNetCalculator

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRISTINE_DIR = BASE_DIR / "structures" / "pristine_mofs"
HER_DIR = BASE_DIR / "structures" / "intermediates_her"
OER_DIR = BASE_DIR / "structures" / "intermediates_oer"
CO2RR_DIR = BASE_DIR / "structures" / "intermediates_co2rr"
OUT_DIR = BASE_DIR / "structures" / "chgnet_prerelaxed"
LOG_DIR = BASE_DIR / "logs"

for d in [OUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

COHORT_CSV = DATA_DIR / "cohort_benchmark_mofs.csv"
CHGNET_RESULTS_CSV = DATA_DIR / "chgnet_screening_results.csv"
LOG_FILE = LOG_DIR / "results.log"

# Optimization thresholds
FMAX_CHGNET = 0.10     # Max residual force (eV/Å) for pre-relaxation
MAX_STEPS = 40         # Cap steps to ensure rapid screening throughput
DEVICE = "cuda"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("script_03_chgnet_screening")


def relax_structure(cif_path: Path, calc, out_cif: Path) -> dict:
    """Relaxes a single structure with CHGNet and returns performance metrics."""
    atoms = ase.io.read(str(cif_path))
    initial_volume = atoms.get_volume()
    atoms.calc = calc

    initial_energy = float(atoms.get_potential_energy())
    initial_forces = atoms.get_forces()
    initial_fmax = float(np.max(np.linalg.norm(initial_forces, axis=1)))

    converged = False
    opt = BFGS(atoms, logfile=None)
    try:
        converged = opt.run(fmax=FMAX_CHGNET, steps=MAX_STEPS)
    except Exception as e:
        logger.warning(f"Optimization exception in {cif_path.name}: {e}")

    final_energy = float(atoms.get_potential_energy())
    final_forces = atoms.get_forces()
    final_fmax = float(np.max(np.linalg.norm(final_forces, axis=1)))
    final_volume = atoms.get_volume()
    volume_change_pct = (final_volume - initial_volume) / initial_volume * 100.0

    # Save pre-relaxed structure
    ase.io.write(str(out_cif), atoms, format="cif")

    return {
        "file": cif_path.name,
        "natoms": len(atoms),
        "initial_energy_eV": initial_energy,
        "relaxed_energy_eV": final_energy,
        "energy_change_eV": final_energy - initial_energy,
        "initial_fmax_eV_A": initial_fmax,
        "final_fmax_eV_A": final_fmax,
        "steps_taken": opt.get_number_of_steps(),
        "converged": converged,
        "volume_change_pct": volume_change_pct
    }


def main():
    logger.info("=== STEP 5: Tier-1 CHGNet Screening & Pre-Relaxation ===")
    
    if not COHORT_CSV.exists():
        logger.error(f"Cohort CSV not found at {COHORT_CSV}")
        sys.exit(1)

    logger.info(f"Initializing CHGNet calculator on device: {DEVICE}...")
    calc = CHGNetCalculator(use_device=DEVICE)

    cohort_df = pd.read_csv(COHORT_CSV)
    qmof_ids = cohort_df["qmof_id"].tolist()
    logger.info(f"Targeting {len(qmof_ids)} MOF cohort systems across 7 intermediate reactions + pristine states.")

    tasks = []
    # Pristine structures
    for q_id in qmof_ids:
        cif_p = PRISTINE_DIR / f"{q_id}.cif"
        if cif_p.exists():
            tasks.append((cif_p, "pristine", "pristine", q_id))

    # Intermediates
    inter_catalog = {
        "her": ["H"],
        "oer": ["OH", "O", "OOH"],
        "co2rr": ["COOH", "CO", "OCHO"]
    }
    dir_map = {
        "her": HER_DIR,
        "oer": OER_DIR,
        "co2rr": CO2RR_DIR
    }

    for rxn, inters in inter_catalog.items():
        src_dir = dir_map[rxn]
        for inter in inters:
            for q_id in qmof_ids:
                cif_i = src_dir / f"{q_id}_{rxn}_{inter}.cif"
                if cif_i.exists():
                    tasks.append((cif_i, rxn, inter, q_id))

    logger.info(f"Total relaxation tasks scheduled: {len(tasks)}")

    results = []
    for idx, (src_cif, rxn, inter, q_id) in enumerate(tasks, 1):
        out_cif = OUT_DIR / src_cif.name
        res = relax_structure(src_cif, calc, out_cif)
        res["qmof_id"] = q_id
        res["reaction"] = rxn
        res["intermediate"] = inter
        results.append(res)
        
        status_str = "CONV" if res["converged"] else "STEP-CAP"
        logger.info(
            f"[{idx:3d}/{len(tasks)}] {src_cif.name:32s} | {status_str} | "
            f"ΔE: {res['energy_change_eV']:+8.3f} eV | fmax: {res['initial_fmax_eV_A']:.2f} -> {res['final_fmax_eV_A']:.2f} eV/Å"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(CHGNET_RESULTS_CSV, index=False)
    logger.info(f"Saved CHGNet screening summary to {CHGNET_RESULTS_CSV}")

    # =========================================================================
    # OUTPUT VALIDATION BLOCK (Required by Protocol Section 9.2 & 11.1)
    # =========================================================================
    validation_passed = True
    validation_errors = []

    # Check 1: Output CSV exists and is populated
    if not CHGNET_RESULTS_CSV.exists() or len(results_df) == 0:
        validation_passed = False
        validation_errors.append(f"Missing or empty results CSV: {CHGNET_RESULTS_CSV}")

    # Check 2: All scheduled tasks completed
    if len(results_df) != len(tasks):
        validation_passed = False
        validation_errors.append(f"Incomplete run: completed {len(results_df)} of {len(tasks)} tasks.")

    # Check 3: Energy descent sanity check (no spontaneous explosive energy increase)
    unstable = results_df[results_df["energy_change_eV"] > 1.0]
    if len(unstable) > 0:
        validation_passed = False
        validation_errors.append(f"Found {len(unstable)} structures with unphysical energy divergence (> +1 eV).")

    if validation_passed:
        logger.info("\n[VALIDATION PASSED] script_03_chgnet_screening.py")
        logger.info(f"Successfully screened and pre-relaxed {len(results_df)} structures via CHGNet.")
    else:
        logger.error("\n[VALIDATION FAILED] script_03_chgnet_screening.py")
        for err in validation_errors:
            logger.error(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
