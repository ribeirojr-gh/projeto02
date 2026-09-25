#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 02: Open Metal Site (OMS) Identification & Stereochemical Adsorbate Placement
Project: mofs-mace-her-oer-co2rr

Reactions & Intermediates Generated:
  - HER:   *H
  - OER:   *OH, *O, *OOH
  - CO2RR: *COOH, *CO, *OCHO

Dependencies:
    pip install pymatgen ase numpy pandas
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import ase.io
from ase import Atom, Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRISTINE_DIR = BASE_DIR / "structures" / "pristine_mofs"
HER_DIR = BASE_DIR / "structures" / "intermediates_her"
OER_DIR = BASE_DIR / "structures" / "intermediates_oer"
CO2RR_DIR = BASE_DIR / "structures" / "intermediates_co2rr"
LOG_DIR = BASE_DIR / "logs"

for d in [HER_DIR, OER_DIR, CO2RR_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

COHORT_CSV = DATA_DIR / "cohort_benchmark_mofs.csv"
SITES_CSV = DATA_DIR / "active_sites_summary.csv"
LOG_FILE = LOG_DIR / "results.log"

# Coordination sphere parameters
COORD_RADIUS = 2.5       # Cutoff to detect coordinating ligands (O, N, Cl, S) around metal (Å)

# Target transition metal set
TRANSITION_METALS = {"Co", "Cu", "Fe", "Mn", "Mo", "Ni", "Ru", "Zn", "Zr"}

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("script_02_site_intermediate_builder")


def get_orthogonal_vector(v: np.ndarray) -> np.ndarray:
    """Find an arbitrary unit vector perpendicular to v."""
    v_norm = v / np.linalg.norm(v)
    cand = np.array([1.0, 0.0, 0.0]) if abs(v_norm[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    ortho = np.cross(v_norm, cand)
    return ortho / np.linalg.norm(ortho)


def detect_active_metal_site(struct: Structure) -> tuple[int, Structure, np.ndarray]:
    """
    Identifies the primary active transition metal site and determines
    its outward-pointing open coordination unit vector (u_open).
    """
    metal_candidates = []
    for idx, site in enumerate(struct):
        if site.specie.symbol in TRANSITION_METALS:
            neighbors = struct.get_neighbors(site, r=COORD_RADIUS)
            metal_candidates.append((idx, site, neighbors))
            
    if not metal_candidates:
        raise ValueError(f"No target transition metal sites found in structure: {struct.composition}")

    # Prioritize site with lowest coordination number (Open Metal Site / CUS)
    metal_candidates.sort(key=lambda item: len(item[2]))
    best_idx, best_site, best_neighbors = metal_candidates[0]

    if len(best_neighbors) == 0:
        u_open = np.array([0.0, 0.0, 1.0])
    else:
        v_ligands = [n.coords - best_site.coords for n in best_neighbors]
        sum_v = np.sum([v / np.linalg.norm(v) for v in v_ligands], axis=0)
        norm_sum = np.linalg.norm(sum_v)
        if norm_sum < 1e-4:
            u_open = np.cross(v_ligands[0], v_ligands[1])
            u_open = u_open / np.linalg.norm(u_open)
        else:
            u_open = -sum_v / norm_sum

    return best_idx, best_site, u_open


def build_adsorbate_species(m_coords: np.ndarray, u_open: np.ndarray, intermediate: str) -> list[tuple[str, np.ndarray]]:
    """
    Returns Cartesian coordinates and elemental symbols for each intermediate.
    """
    u = u_open / np.linalg.norm(u_open)
    v_ortho = get_orthogonal_vector(u)

    species = []

    if intermediate == "H":
        # HER: *H
        species.append(("H", m_coords + 1.55 * u))

    elif intermediate == "OH":
        # OER: *OH
        r_o = m_coords + 1.85 * u
        theta = np.radians(109.5)
        h_dir = np.cos(theta) * u + np.sin(theta) * v_ortho
        species.append(("O", r_o))
        species.append(("H", r_o + 0.97 * h_dir))

    elif intermediate == "O":
        # OER: *O (oxo)
        species.append(("O", m_coords + 1.68 * u))

    elif intermediate == "OOH":
        # OER: *OOH
        r_o1 = m_coords + 1.85 * u
        theta1 = np.radians(110.0)
        o2_dir = np.cos(theta1) * u + np.sin(theta1) * v_ortho
        r_o2 = r_o1 + 1.45 * o2_dir
        theta2 = np.radians(105.0)
        h_dir = np.cos(theta2) * o2_dir + np.sin(theta2) * v_ortho
        species.append(("O", r_o1))
        species.append(("O", r_o2))
        species.append(("H", r_o2 + 0.98 * h_dir))

    elif intermediate == "CO":
        # CO2RR: *CO
        r_c = m_coords + 1.85 * u
        r_o = r_c + 1.15 * u
        species.append(("C", r_c))
        species.append(("O", r_o))

    elif intermediate == "COOH":
        # CO2RR: *COOH (C-bound carboxyl)
        r_c = m_coords + 1.95 * u
        theta_c_o = np.radians(120.0)
        dir_o1 = np.cos(theta_c_o) * u + np.sin(theta_c_o) * v_ortho
        r_o1 = r_c + 1.22 * dir_o1
        dir_o2 = np.cos(theta_c_o) * u - np.sin(theta_c_o) * v_ortho
        r_o2 = r_c + 1.35 * dir_o2
        r_h = r_o2 + 0.97 * dir_o2
        species.append(("C", r_c))
        species.append(("O", r_o1))
        species.append(("O", r_o2))
        species.append(("H", r_h))

    elif intermediate == "OCHO":
        # CO2RR: *OCHO (formate, O-bound)
        r_o1 = m_coords + 1.95 * u
        theta_o_c = np.radians(120.0)
        dir_c = np.cos(theta_o_c) * u + np.sin(theta_o_c) * v_ortho
        r_c = r_o1 + 1.30 * dir_c
        r_o2 = r_c + 1.22 * u
        r_h = r_c + 1.09 * v_ortho
        species.append(("O", r_o1))
        species.append(("C", r_c))
        species.append(("O", r_o2))
        species.append(("H", r_h))

    else:
        raise ValueError(f"Unknown intermediate: {intermediate}")

    return species


def main():
    logger.info("=== STEP 4: Open Metal Site Detection & Adsorbate Placement ===")
    
    if not COHORT_CSV.exists():
        logger.error(f"Cohort CSV not found: {COHORT_CSV}")
        sys.exit(1)

    cohort_df = pd.read_csv(COHORT_CSV)
    logger.info(f"Loaded benchmark cohort with {len(cohort_df)} MOF systems.")

    intermediates_catalog = {
        "HER": ["H"],
        "OER": ["OH", "O", "OOH"],
        "CO2RR": ["COOH", "CO", "OCHO"]
    }

    dir_map = {
        "HER": HER_DIR,
        "OER": OER_DIR,
        "CO2RR": CO2RR_DIR
    }

    site_records = []
    generated_files = []

    for _, row in cohort_df.iterrows():
        qmof_id = row["qmof_id"]
        cif_path = PRISTINE_DIR / f"{qmof_id}.cif"
        if not cif_path.exists():
            logger.warning(f"Structure {cif_path} does not exist, skipping.")
            continue

        struct = Structure.from_file(str(cif_path))
        try:
            site_idx, m_site, u_open = detect_active_metal_site(struct)
            cn = len(struct.get_neighbors(m_site, r=COORD_RADIUS))
            logger.info(f"System: {qmof_id} | Metal: {m_site.specie.symbol} (site #{site_idx}) | Coord. Number: {cn} | u_open: {np.round(u_open, 3)}")

            site_records.append({
                "qmof_id": qmof_id,
                "metal": m_site.specie.symbol,
                "site_index": site_idx,
                "coordination_number": cn,
                "open_vector_x": u_open[0],
                "open_vector_y": u_open[1],
                "open_vector_z": u_open[2]
            })

            # Base ASE Atoms object
            base_atoms = AseAtomsAdaptor.get_atoms(struct)
            m_coords = base_atoms[site_idx].position

            # Generate each reaction intermediate using ASE
            for rxn, inters in intermediates_catalog.items():
                out_dir = dir_map[rxn]
                for inter in inters:
                    inter_atoms = base_atoms.copy()
                    adsorbate_species = build_adsorbate_species(m_coords, u_open, inter)
                    
                    for elem, coords in adsorbate_species:
                        inter_atoms.append(Atom(elem, position=coords))

                    out_cif = out_dir / f"{qmof_id}_{rxn.lower()}_{inter}.cif"
                    ase.io.write(str(out_cif), inter_atoms, format="cif")
                    generated_files.append((out_cif, site_idx, len(inter_atoms) - len(adsorbate_species)))

        except Exception as e:
            logger.error(f"Error processing {qmof_id}: {e}")

    # Export active sites summary table
    sites_df = pd.DataFrame(site_records)
    sites_df.to_csv(SITES_CSV, index=False)
    logger.info(f"Saved active sites summary to {SITES_CSV}")
    logger.info(f"Total intermediate structures generated: {len(generated_files)}")

    # =========================================================================
    # OUTPUT VALIDATION BLOCK (Required by Protocol Section 9.2 & 11.1)
    # =========================================================================
    validation_passed = True
    validation_errors = []

    # Check 1: Active site table exists
    if not SITES_CSV.exists() or len(sites_df) == 0:
        validation_passed = False
        validation_errors.append(f"Missing or empty active sites table: {SITES_CSV}")

    # Check 2: Expected files count (18 systems * 7 intermediates = 126 structures)
    expected_count = len(cohort_df) * 7
    if len(generated_files) != expected_count:
        validation_passed = False
        validation_errors.append(f"Mismatch in generated intermediates: expected {expected_count}, got {len(generated_files)}")

    # Check 3: Distance verification for all generated intermediate structures
    for fpath, m_idx, ads_first_idx in generated_files:
        test_atoms = ase.io.read(str(fpath))
        d_bind = test_atoms.get_distance(m_idx, ads_first_idx, mic=True)
        if d_bind > 2.6 or d_bind < 1.2:
            validation_passed = False
            validation_errors.append(f"Unphysical binding distance in {fpath.name}: {d_bind:.2f} Å (metal idx {m_idx} -> ads idx {ads_first_idx})")
            break

    if validation_passed:
        logger.info("\n[VALIDATION PASSED] script_02_site_intermediate_builder.py")
        logger.info(f"Successfully generated and validated {len(generated_files)} intermediate structures with physical coordination bonds.")
    else:
        logger.error("\n[VALIDATION FAILED] script_02_site_intermediate_builder.py")
        for err in validation_errors:
            logger.error(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
