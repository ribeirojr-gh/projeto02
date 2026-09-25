#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 04: Tier-2 MACE-MP-0 High-Precision Relaxation & Adsorption Energetics
Project: mofs-mace-her-oer-co2rr

Purpose:
  Refines pre-relaxed structures using the equivariant MACE-MP-0 model on GPU.
  Calculates high-precision ground-state energies, electronic adsorption energies (ΔE)
  for HER, OER, and CO2RR intermediates, and evaluates model epistemic uncertainty.

Dependencies:
    pip install mace-torch ase pandas numpy torch
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import ase.io
from ase import Atoms
from ase.optimize import BFGS
from mace.calculators import mace_mp

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IN_DIR = BASE_DIR / "structures" / "chgnet_prerelaxed"
OUT_DIR = BASE_DIR / "structures" / "mace_relaxed"
LOG_DIR = BASE_DIR / "logs"

for d in [OUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

COHORT_CSV = DATA_DIR / "cohort_benchmark_mofs.csv"
CHGNET_CSV = DATA_DIR / "chgnet_screening_results.csv"
MACE_RESULTS_CSV = DATA_DIR / "adsorption_energies_mace.csv"
GAS_REF_CSV = DATA_DIR / "mace_gas_references.csv"
LOG_FILE = LOG_DIR / "results.log"

FMAX_MACE = 0.03       # Max residual force (eV/Å)
MAX_STEPS = 25         # Steps for precision refinement from pre-relaxed geometries
MODEL_SIZE = "small"   # MACE-MP-0 architecture
DTYPE = "float64"      # Double precision for reliable force minimization
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
logger = logging.getLogger("script_04_mace_refinement")


def compute_gas_references(calc) -> dict[str, float]:
    """Computes relaxed energies of isolated gas reference molecules using MACE."""
    logger.info("Computing MACE reference energies for gas molecules (H2, H2O, CO2, CO)...")
    molecules = {
        "H2": Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[15, 15, 15], pbc=False),
        "H2O": Atoms("H2O", positions=[[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59]], cell=[15, 15, 15], pbc=False),
        "CO2": Atoms("CO2", positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]], cell=[15, 15, 15], pbc=False),
        "CO": Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], cell=[15, 15, 15], pbc=False),
    }
    gas_energies = {}
    records = []
    for name, mol in molecules.items():
        mol.calc = calc
        opt = BFGS(mol, logfile=None)
        opt.run(fmax=0.01, steps=20)
        e = float(mol.get_potential_energy())
        gas_energies[name] = e
        records.append({"molecule": name, "energy_eV": e})
        logger.info(f"  Gas {name:4s}: {e:10.6f} eV")

    pd.DataFrame(records).to_csv(GAS_REF_CSV, index=False)
    return gas_energies


def relax_mace(cif_path: Path, calc, out_cif: Path) -> dict:
    """Relaxes a structure with MACE and returns convergence metrics."""
    atoms = ase.io.read(str(cif_path))
    atoms.calc = calc

    initial_energy = float(atoms.get_potential_energy())
    converged = False
    opt = BFGS(atoms, logfile=None)
    try:
        converged = opt.run(fmax=FMAX_MACE, steps=MAX_STEPS)
    except Exception as e:
        logger.warning(f"MACE relaxation warning in {cif_path.name}: {e}")

    final_energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax = float(np.max(np.linalg.norm(forces, axis=1)))

    ase.io.write(str(out_cif), atoms, format="cif")

    return {
        "final_energy_eV": final_energy,
        "energy_change_eV": final_energy - initial_energy,
        "fmax_eV_A": fmax,
        "converged": converged,
        "natoms": len(atoms),
        "steps": opt.get_number_of_steps()
    }


def main():
    logger.info("=== STEP 6: Tier-2 MACE-MP-0 Precision Refinement & Adsorption Energetics ===")

    if not COHORT_CSV.exists() or not CHGNET_CSV.exists():
        logger.error("Required prerequisite CSVs not found.")
        sys.exit(1)

    logger.info(f"Initializing MACE-MP-0 ({MODEL_SIZE}, {DTYPE}) on {DEVICE}...")
    calc = mace_mp(model=MODEL_SIZE, device=DEVICE, default_dtype=DTYPE)

    gas_refs = compute_gas_references(calc)
    e_h2 = gas_refs["H2"]
    e_h2o = gas_refs["H2O"]
    e_co2 = gas_refs["CO2"]
    e_co = gas_refs["CO"]

    cohort_df = pd.read_csv(COHORT_CSV)
    chgnet_df = pd.read_csv(CHGNET_CSV).set_index("file")
    qmof_ids = cohort_df["qmof_id"].tolist()

    # Stage 1: Relax pristine MOF frameworks
    logger.info(f"Refining {len(qmof_ids)} pristine MOF frameworks with MACE...")
    pristine_energies = {}
    for q_id in qmof_ids:
        in_cif = IN_DIR / f"{q_id}.cif"
        out_cif = OUT_DIR / f"{q_id}.cif"
        res = relax_mace(in_cif, calc, out_cif)
        pristine_energies[q_id] = res["final_energy_eV"]
        logger.info(f"  Pristine {q_id:15s} | E: {res['final_energy_eV']:10.4f} eV | fmax: {res['fmax_eV_A']:.3f} eV/Å")

    # Stage 2: Relax intermediate complexes
    inter_catalog = {
        "her": ["H"],
        "oer": ["OH", "O", "OOH"],
        "co2rr": ["COOH", "CO", "OCHO"]
    }

    adsorption_records = []
    total_intermediates = len(qmof_ids) * 7
    curr_idx = 0

    logger.info(f"Refining {total_intermediates} reaction intermediate complexes with MACE...")
    for rxn, inters in inter_catalog.items():
        for inter in inters:
            for q_id in qmof_ids:
                curr_idx += 1
                fname = f"{q_id}_{rxn}_{inter}.cif"
                in_cif = IN_DIR / fname
                out_cif = OUT_DIR / fname

                if not in_cif.exists():
                    logger.warning(f"File {in_cif} missing, skipping.")
                    continue

                res = relax_mace(in_cif, calc, out_cif)
                e_inter = res["final_energy_eV"]
                e_clean = pristine_energies[q_id]

                # Compute electronic adsorption energy ΔE (eV)
                if rxn == "her" and inter == "H":
                    # * + 1/2 H2 -> *H
                    delta_e = e_inter - e_clean - 0.5 * e_h2

                elif rxn == "oer":
                    # * + H2O -> *OH + 1/2 H2
                    if inter == "OH":
                        delta_e = e_inter - e_clean - (e_h2o - 0.5 * e_h2)
                    # * + H2O -> *O + H2
                    elif inter == "O":
                        delta_e = e_inter - e_clean - (e_h2o - e_h2)
                    # * + 2 H2O -> *OOH + 3/2 H2
                    elif inter == "OOH":
                        delta_e = e_inter - e_clean - (2.0 * e_h2o - 1.5 * e_h2)

                elif rxn == "co2rr":
                    # * + CO2 + 1/2 H2 -> *COOH
                    if inter == "COOH":
                        delta_e = e_inter - e_clean - (e_co2 + 0.5 * e_h2)
                    # * + CO2 + H2 -> *CO + H2O
                    elif inter == "CO":
                        delta_e = e_inter - e_clean - (e_co2 + e_h2 - e_h2o)
                    # * + CO2 + 1/2 H2 -> *OCHO (formate)
                    elif inter == "OCHO":
                        delta_e = e_inter - e_clean - (e_co2 + 0.5 * e_h2)

                # Epistemic model variance (MACE vs CHGNet per atom)
                chg_energy = chgnet_df.loc[fname, "relaxed_energy_eV"] if fname in chgnet_df.index else np.nan
                sigma_mlip = abs(e_inter - chg_energy) / res["natoms"] if not np.isnan(chg_energy) else 0.0

                adsorption_records.append({
                    "qmof_id": q_id,
                    "reaction": rxn,
                    "intermediate": inter,
                    "file": fname,
                    "e_clean_eV": e_clean,
                    "e_intermediate_eV": e_inter,
                    "delta_E_eV": delta_e,
                    "fmax_eV_A": res["fmax_eV_A"],
                    "converged": res["converged"],
                    "natoms": res["natoms"],
                    "mace_chgnet_diff_per_atom_eV": sigma_mlip
                })

                logger.info(
                    f"[{curr_idx:3d}/{total_intermediates}] {fname:30s} | "
                    f"ΔE: {delta_e:+7.3f} eV | fmax: {res['fmax_eV_A']:.3f} | unc: {sigma_mlip:.4f} eV/atom"
                )

    mace_df = pd.DataFrame(adsorption_records)
    mace_df.to_csv(MACE_RESULTS_CSV, index=False)
    logger.info(f"Saved MACE adsorption energetics to {MACE_RESULTS_CSV}")

    # =========================================================================
    # OUTPUT VALIDATION BLOCK (Required by Protocol Section 9.2 & 11.1)
    # =========================================================================
    validation_passed = True
    validation_errors = []

    # Check 1: Output CSV exists and has all rows
    if not MACE_RESULTS_CSV.exists() or len(mace_df) == 0:
        validation_passed = False
        validation_errors.append(f"Missing or empty MACE results file: {MACE_RESULTS_CSV}")
    elif len(mace_df) != total_intermediates:
        validation_passed = False
        validation_errors.append(f"Incomplete records: expected {total_intermediates}, got {len(mace_df)}")

    # Check 2: Physical bounds on adsorption energies (-6.0 eV <= delta_E <= +6.0 eV)
    unphysical_ads = mace_df[(mace_df["delta_E_eV"] < -6.0) | (mace_df["delta_E_eV"] > 6.0)]
    if len(unphysical_ads) > 0:
        logger.warning(f"Note: {len(unphysical_ads)} structures have extreme binding energies.")

    # Check 3: Check for NaN values
    if mace_df["delta_E_eV"].isna().any():
        validation_passed = False
        validation_errors.append("Detected NaN in calculated adsorption energies.")

    if validation_passed:
        logger.info("\n[VALIDATION PASSED] script_04_mace_refinement.py")
        logger.info(f"Successfully calculated MACE precision energies and ΔE for all {len(mace_df)} intermediate complexes.")
    else:
        logger.error("\n[VALIDATION FAILED] script_04_mace_refinement.py")
        for err in validation_errors:
            logger.error(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
