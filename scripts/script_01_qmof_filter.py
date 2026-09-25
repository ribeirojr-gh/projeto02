#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 01: QMOF Database Ingestion, Electrocatalytic Screening & Structure Extraction
Project: mofs-mace-her-oer-co2rr

Dependencies:
    pip install pymatgen pandas
"""

import os
import sys
import io
import zipfile
import logging
from pathlib import Path
import pandas as pd
from pymatgen.core import Structure, Composition

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STRUCTURES_DIR = BASE_DIR / "structures" / "pristine_mofs"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = DATA_DIR / "qmof_database" / "qmof.csv"
ZIP_FILE = DATA_DIR / "qmof_database.zip"
FILTERED_CSV = DATA_DIR / "metadata_qmof_filtered.csv"
COHORT_CSV = DATA_DIR / "cohort_benchmark_mofs.csv"
LOG_FILE = LOG_DIR / "results.log"

# Target Transition Metals of interest for HER, OER, and CO2RR
TARGET_METALS = ["Co", "Cu", "Fe", "Mn", "Mo", "Ni", "Ru", "Zn", "Zr"]

# Physicochemical filter thresholds
FILTER_SYNTHESIZED_ONLY = True
MIN_PLD = 2.5       # Pore Limiting Diameter (Å) to allow H2O (2.65 Å) and CO2 (3.3 Å) diffusion
MAX_NATOMS = 200    # Atom count limit for tractable MLIP relaxation & localized PHVA thermochemistry
TOP_PER_METAL = 2   # Representative benchmark structures per transition metal family

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("script_01_qmof_filter")


def get_metals(formula_str: str) -> list[str]:
    """Extract transition metals present in the formula using Pymatgen."""
    try:
        comp = Composition(formula_str)
        return [el.symbol for el in comp.elements if el.symbol in TARGET_METALS]
    except Exception:
        return []


def main():
    logger.info("=== STEP 2: Database Search & Filtering (QMOF / Materials Project MOF Explorer) ===")
    
    if not CSV_FILE.exists():
        logger.error(f"QMOF database CSV not found at {CSV_FILE}")
        sys.exit(1)
        
    logger.info(f"Loading QMOF database from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE, low_memory=False)
    total_structures = len(df)
    logger.info(f"Loaded {total_structures} total MOF structures from QMOF Database.")

    # Apply chemical identification
    logger.info("Identifying active metal nodes in framework compositions...")
    df["target_metals"] = df["info.formula"].apply(get_metals)
    df["primary_metal"] = df["target_metals"].apply(lambda m: m[0] if len(m) > 0 else None)
    df["has_target_metal"] = df["target_metals"].apply(lambda m: len(m) > 0)

    # Filtering pipeline
    condition = (df["has_target_metal"])
    if FILTER_SYNTHESIZED_ONLY:
        condition &= (df["info.synthesized"] == True)
    condition &= (df["info.pld"] >= MIN_PLD)
    condition &= (df["info.natoms"] <= MAX_NATOMS)

    filtered_df = df[condition].copy()
    logger.info(f"Filtered candidate pool: {len(filtered_df)} structures meeting all physical/chemical criteria:")
    logger.info(f"  - Synthesized experimentally: {FILTER_SYNTHESIZED_ONLY}")
    logger.info(f"  - Pore Limiting Diameter (PLD) >= {MIN_PLD} Å")
    logger.info(f"  - Unit Cell Size (N_atoms) <= {MAX_NATOMS}")
    
    for metal in sorted(TARGET_METALS):
        count = filtered_df["target_metals"].apply(lambda m_list: metal in m_list).sum()
        logger.info(f"  * {metal}-based MOF candidates: {count}")

    # Export filtered database table
    filtered_df.to_csv(FILTERED_CSV, index=False)
    logger.info(f"Saved filtered screening metadata to {FILTERED_CSV}")

    # Select representative benchmark cohort across metal families
    logger.info(f"Selecting top {TOP_PER_METAL} representative MOF archetypes per metal family for benchmark cohort...")
    cohort_list = []
    for metal in sorted(TARGET_METALS):
        metal_candidates = filtered_df[filtered_df["primary_metal"] == metal].copy()
        if len(metal_candidates) > 0:
            # Sort by density and pore diameter to pick well-characterized porous crystals
            metal_candidates = metal_candidates.sort_values(
                by=["info.pld", "outputs.pbe.energy_total"], ascending=[False, True]
            )
            selected = metal_candidates.head(TOP_PER_METAL)
            cohort_list.append(selected)

    cohort_df = pd.concat(cohort_list, ignore_index=True)
    cohort_df.to_csv(COHORT_CSV, index=False)
    logger.info(f"Constructed benchmark cohort of {len(cohort_df)} MOF systems across {len(cohort_list)} metal families:")
    for _, row in cohort_df.iterrows():
        logger.info(f"  ID: {row['qmof_id']} | Metal: {row['primary_metal']} | Formula: {row['info.formula']} | SpaceGroup: {row['info.symmetry.spacegroup']} | PLD: {row['info.pld']:.2f} Å | Atoms: {row['info.natoms']}")

    # Extract CIF structures from zip archive
    logger.info("Extracting pristine CIF structures for the benchmark cohort from QMOF archive...")
    extracted_cifs = []
    with zipfile.ZipFile(ZIP_FILE, "r") as root_z:
        with root_z.open("qmof_database/relaxed_structures.zip") as sub_z_bytes:
            with zipfile.ZipFile(io.BytesIO(sub_z_bytes.read())) as sub_z:
                for qmof_id in cohort_df["qmof_id"]:
                    cif_name = f"relaxed_structures/{qmof_id}.cif"
                    out_path = STRUCTURES_DIR / f"{qmof_id}.cif"
                    try:
                        cif_content = sub_z.read(cif_name)
                        with open(out_path, "wb") as f_out:
                            f_out.write(cif_content)
                        # Validate structure parsing via pymatgen
                        struct = Structure.from_file(str(out_path))
                        extracted_cifs.append(out_path)
                        logger.info(f"  ✓ Extracted & validated: {qmof_id}.cif (Volume: {struct.volume:.2f} Å³, Atoms: {len(struct)})")
                    except Exception as e:
                        logger.error(f"  ✗ Failed to extract or validate {qmof_id}: {e}")

    # =========================================================================
    # OUTPUT VALIDATION BLOCK (Required by Protocol Section 9.2 & 11.1)
    # =========================================================================
    validation_passed = True
    validation_errors = []

    # Check 1: Filtered CSV exists and is non-empty
    if not FILTERED_CSV.exists() or FILTERED_CSV.stat().st_size == 0:
        validation_passed = False
        validation_errors.append(f"Missing or empty filtered metadata CSV: {FILTERED_CSV}")
    elif len(filtered_df) < 100:
        validation_passed = False
        validation_errors.append(f"Filtered candidate pool unexpectedly small ({len(filtered_df)} entries)")

    # Check 2: Cohort CSV exists and matches expected count
    if not COHORT_CSV.exists() or len(cohort_df) == 0:
        validation_passed = False
        validation_errors.append("Benchmark cohort CSV is missing or empty.")

    # Check 3: All CIF files extracted and verified
    if len(extracted_cifs) != len(cohort_df):
        validation_passed = False
        validation_errors.append(f"Mismatch in extracted CIF count: expected {len(cohort_df)}, extracted {len(extracted_cifs)}")

    if validation_passed:
        logger.info("\n[VALIDATION PASSED] script_01_qmof_filter.py")
        logger.info(f"Successfully processed {len(filtered_df)} candidate MOFs and extracted {len(extracted_cifs)} benchmark structures.")
    else:
        logger.error("\n[VALIDATION FAILED] script_01_qmof_filter.py")
        for err in validation_errors:
            logger.error(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
