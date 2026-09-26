#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 08: Local DFT Benchmark Validation via GPAW (PBE/LCAO)
Project: mofs-mace-her-oer-co2rr (Repo: projeto02-26092026)

Purpose:
  Performs first-principles Density Functional Theory (DFT) calculations
  using GPAW on local CPU cores. Validates resource availability (RAM/CPUs)
  before execution and benchmarks MACE-MP-0 energies against explicit DFT.

Dependencies:
    pip install gpaw ase pandas numpy psutil
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
from ase import Atoms
import ase.io
from gpaw import GPAW

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_CSV = DATA_DIR / "gpaw_dft_benchmark.csv"
LOG_FILE = LOG_DIR / "results.log"

# Minimum system resource safety thresholds
MIN_AVAILABLE_RAM_GB = 4.0
XC_FUNCTIONAL = "PBE"
MODE = "lcao"          # Fast, accurate linear combination of atomic orbitals

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("script_08_gpaw_dft_validation")


def check_system_resources():
    """Verifies available RAM and CPU before launching DFT."""
    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024 ** 3)
    cpu_count = psutil.cpu_count(logical=True)
    load_avg = os.getloadavg()[0]

    logger.info(f"System Resource Evaluation:")
    logger.info(f"  Available RAM: {available_gb:.2f} GB (Total: {vm.total / (1024**3):.2f} GB)")
    logger.info(f"  CPU Cores: {cpu_count} | 1-min Load Average: {load_avg:.2f}")

    if available_gb < MIN_AVAILABLE_RAM_GB:
        raise ResourceWarning(
            f"Insufficient available RAM: {available_gb:.2f} GB < {MIN_AVAILABLE_RAM_GB} GB threshold."
        )


def run_gpaw_molecule(atoms: Atoms, name: str, txt_name: str) -> float:
    """Runs a GPAW calculation on a centered molecular system."""
    atoms_copy = atoms.copy()
    atoms_copy.center(vacuum=3.5)
    calc = GPAW(mode=MODE, xc=XC_FUNCTIONAL, txt=txt_name)
    atoms_copy.calc = calc
    energy = float(atoms_copy.get_potential_energy())
    logger.info(f"  GPAW ({XC_FUNCTIONAL}/{MODE}) {name:8s}: {energy:10.4f} eV")
    return energy


def main():
    logger.info("=== STEP 10: Local DFT Benchmark Validation via GPAW ===")

    # 1. Resource Pre-Check
    try:
        check_system_resources()
    except Exception as e:
        logger.error(f"Resource check failed: {e}")
        sys.exit(1)

    # 2. Benchmark Reference Systems
    test_systems = {
        "H2": Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], pbc=False),
        "CO": Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], pbc=False),
        "H2O": Atoms("H2O", positions=[[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59]], pbc=False),
        "CO2": Atoms("CO2", positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]], pbc=False),
    }

    records = []
    # MACE energies from Step 6 for cross-validation
    mace_refs = {
        "H2": -6.5577,
        "H2O": -14.0515,
        "CO2": -22.8299,
        "CO": -14.3862
    }

    logger.info("Computing explicit DFT ground-state energies with GPAW...")
    for name, atoms in test_systems.items():
        txt_log = str(LOG_DIR / f"gpaw_{name}.txt")
        try:
            e_dft = run_gpaw_molecule(atoms, name, txt_log)
            e_mace = mace_refs.get(name, np.nan)
            diff = e_mace - e_dft if not np.isnan(e_mace) else np.nan

            records.append({
                "system": name,
                "natoms": len(atoms),
                "e_gpaw_dft_eV": e_dft,
                "e_mace_eV": e_mace,
                "diff_mace_dft_eV": diff,
                "diff_per_atom_eV": diff / len(atoms) if not np.isnan(diff) else np.nan
            })
        except Exception as e:
            logger.error(f"GPAW calculation failed for {name}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(BENCHMARK_CSV, index=False)
    logger.info(f"Saved GPAW DFT benchmark results to {BENCHMARK_CSV}")

    # =========================================================================
    # OUTPUT VALIDATION BLOCK (Required by Protocol Section 9.2 & 11.1)
    # =========================================================================
    validation_passed = True
    validation_errors = []

    if not BENCHMARK_CSV.exists() or len(df) == 0:
        validation_passed = False
        validation_errors.append(f"Missing or empty GPAW benchmark CSV: {BENCHMARK_CSV}")

    if df["e_gpaw_dft_eV"].isna().any():
        validation_passed = False
        validation_errors.append("Detected NaN in GPAW DFT ground-state energy outputs.")

    if validation_passed:
        logger.info("\n[VALIDATION PASSED] script_08_gpaw_dft_validation.py")
        logger.info(f"Successfully validated local DFT calculations on {len(df)} systems.")
    else:
        logger.error("\n[VALIDATION FAILED] script_08_gpaw_dft_validation.py")
        for err in validation_errors:
            logger.error(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
