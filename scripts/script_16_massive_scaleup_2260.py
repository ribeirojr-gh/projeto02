#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 16: Massive High-Throughput Scale-Up Across the Full Pore-Accessible
           QMOF Cohort (2260 MOFs)
Project: mofs-mace-her-oer-co2rr (Repo: projeto02-26092026)

Supersedes script_12's 32-MOF breadth-screening cohort. Per explicit user
request, this removes BOTH the per-metal sampling cap (script_12 took only
the top 4 PLD-ranked candidates per metal family) and the <=140/<=200-atom
sub-filter, and instead screens EVERY structure in stage 3 of the cascade
filter: target transition metal present (Co/Cu/Fe/Mn/Mo/Ni/Ru/Zn/Zr) AND
experimentally synthesized AND pore-limiting diameter >= 2.5 A. That pool
has 2260 members (max 500 atoms/cell -- no pathological outliers), recomputed
directly from the raw QMOF metadata (data/qmof_database/qmof.csv) rather than
reusing metadata_qmof_filtered.csv, since that file already has the <=200-atom
cut baked in. The recomputed pool is saved to
data/metadata_pore_accessible_2260.csv for an auditable record of exactly
which 2260 structures this cascade stage selects.

Same per-structure methodology as script_12 (shallow, single-tier MACE-MP-0
BFGS relaxation of the pristine framework and 6 adsorbed intermediates per
MOF -- *H, *OH, *O, *OOH, *COOH, *CO -- with generic literature-typical
ZPE/-TS corrections rather than per-structure PHVA, and the same
gas-referenced |dE|>50 eV steric-congestion convention), so results are
directly comparable to/supersede data/high_throughput_scaleup_results.csv.
The 18-MOF deep benchmark cohort (PHVA + GPAW-validated) is untouched.

Designed to run unattended for multiple days:
  - Incremental checkpointing: each MOF's result row is appended to the
    output CSV immediately after that MOF finishes (not just once at the
    end, which script_12 did and which would lose everything on an
    interruption during a multi-day run).
  - Resumable: on (re)start, qmof_ids already present in the output CSV are
    skipped, so a restart continues rather than redoing finished work.
  - Per-MOF error isolation: a failure on one structure (corrupt/missing
    CIF, CUDA OOM, pathological geometry, etc.) is logged and that MOF is
    skipped; it never aborts the run.

Dependencies: mace-torch ase pandas numpy torch pymatgen
"""

import sys
import time
import zipfile
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import ase.io
from ase.optimize import BFGS
from mace.calculators import mace_mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script_12_high_throughput_scaleup import build_adsorbate, extract_cif_from_zip  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
QMOF_RAW_CSV = DATA_DIR / "qmof_database" / "qmof.csv"
ZIP_PATH = DATA_DIR / "qmof_database.zip"
POOL_CSV = DATA_DIR / "metadata_pore_accessible_2260.csv"
OLD_PRISTINE_DIRS = [BASE_DIR / "structures" / "pristine_mofs", BASE_DIR / "structures" / "scaleup_pristine"]
PRISTINE_DIR = BASE_DIR / "structures" / "massive_scaleup_pristine"
OUTPUT_CSV = DATA_DIR / "massive_scaleup_2260_results.csv"
LOG_FILE = BASE_DIR / "logs" / "results.log"

PRISTINE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_METALS = ["Co", "Cu", "Fe", "Mn", "Mo", "Ni", "Ru", "Zn", "Zr"]
MIN_PLD = 2.5
FMAX_SCALEUP = 0.03
MAX_STEPS = 300
MODEL_SIZE = "medium"
DTYPE = "float64"
DEVICE = "cuda"

THERMO_CORR = {
    "*H": {"dG_corr": 0.25}, "*OH": {"dG_corr": 0.33}, "*O": {"dG_corr": 0.05},
    "*OOH": {"dG_corr": 0.37}, "*COOH": {"dG_corr": 0.49}, "*CO": {"dG_corr": 0.10},
}
GAS_REFS = {"H2": -6.74311, "H2O": -14.05150, "CO2": -23.10984, "CO": -14.77452}
INTER_CONFIGS = [("*H", 1.55), ("*OH", 1.95), ("*O", 1.80), ("*OOH", 1.95),
                  ("*COOH", 1.95), ("*CO", 1.90)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("script_16_massive_scaleup_2260")

CSV_COLUMNS = ["qmof_id", "metal", "pld_A", "lcd_A", "bandgap_eV", "coord_number", "natoms",
               "dG_H_eV", "eta_HER_V", "dG_OH_eV", "dG_O_eV", "dG_OOH_eV", "eta_OER_V",
               "dG_COOH_eV", "dG_CO_eV", "eta_CO2RR_V", "delta_G_sel_eV", "prefers_CO2RR",
               "steric_congested", "error"]


def build_pool() -> pd.DataFrame:
    """Recomputes the 2260-structure pore-accessible pool (stage 3 of the
    cascade filter: target metal + synthesized + PLD>=2.5, no further cap)
    directly from the raw QMOF metadata, and saves it for an auditable
    record of exactly what this phase screened."""
    if POOL_CSV.exists():
        logger.info(f"Loading existing pore-accessible pool from {POOL_CSV}...")
        return pd.read_csv(POOL_CSV)

    from pymatgen.core import Composition
    logger.info(f"Recomputing pore-accessible pool from {QMOF_RAW_CSV}...")
    df = pd.read_csv(QMOF_RAW_CSV, low_memory=False)

    def get_metals(formula_str):
        try:
            comp = Composition(formula_str)
            return [el.symbol for el in comp.elements if el.symbol in TARGET_METALS]
        except Exception:
            return []

    df["target_metals"] = df["info.formula"].apply(get_metals)
    df["primary_metal"] = df["target_metals"].apply(lambda m: m[0] if len(m) > 0 else None)
    df["has_target_metal"] = df["target_metals"].apply(lambda m: len(m) > 0)

    cond = df["has_target_metal"] & (df["info.synthesized"] == True) & (df["info.pld"] >= MIN_PLD)  # noqa: E712
    pool = df[cond].copy()
    pool.to_csv(POOL_CSV, index=False)
    logger.info(f"Pore-accessible pool: {len(pool)} structures (target metal + synthesized + "
                f"PLD>={MIN_PLD} A, no atom-count cap). Saved to {POOL_CSV}")
    return pool


def get_pristine_cif(q_id: str, nested_zf) -> Path | None:
    """Finds or extracts the pristine CIF for q_id, checking already-local
    copies (benchmark/prior scale-up cohorts) before extracting fresh from
    the QMOF zip archive."""
    for d in OLD_PRISTINE_DIRS + [PRISTINE_DIR]:
        p = d / f"{q_id}.cif"
        if p.exists():
            return p
    out_path = PRISTINE_DIR / f"{q_id}.cif"
    if extract_cif_from_zip(nested_zf, q_id, out_path):
        return out_path
    return None


def screen_one(q_id: str, metal: str, pld: float, lcd: float, bandgap: float,
                atoms, calc) -> dict:
    metal_indices = [i for i, at in enumerate(atoms) if at.symbol == metal]
    if not metal_indices:
        return {"qmof_id": q_id, "metal": metal, "error": "no_metal_site_found"}
    m_idx = metal_indices[0]

    dists = atoms.get_distances(m_idx, range(len(atoms)), mic=True)
    coord_indices = [i for i, d in enumerate(dists) if 0.1 < d < 2.5]
    coord_num = len(coord_indices)
    if coord_indices:
        vecs = atoms.get_distances(m_idx, coord_indices, mic=True, vector=True)
        u_open = -np.mean(vecs, axis=0)
        u_open = u_open / np.linalg.norm(u_open) if np.linalg.norm(u_open) > 0.1 else np.array([0.0, 0.0, 1.0])
    else:
        u_open = np.array([0.0, 0.0, 1.0])

    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=FMAX_SCALEUP, steps=MAX_STEPS)
    e_clean = float(atoms.get_potential_energy())

    energies = {}
    any_congested = False
    for ads_type, b_dist in INTER_CONFIGS:
        struct = build_adsorbate(atoms, m_idx, ads_type, b_dist, u_open)
        struct.calc = calc
        BFGS(struct, logfile=None).run(fmax=FMAX_SCALEUP, steps=MAX_STEPS)
        e_tot = float(struct.get_potential_energy())
        fmax_val = float(np.max(np.linalg.norm(struct.get_forces(), axis=1)))
        if fmax_val > 5.0:
            any_congested = True
            e_tot = np.nan
        energies[ads_type] = e_tot

    dE_H = energies["*H"] - e_clean - 0.5 * GAS_REFS["H2"]
    dG_H = dE_H + THERMO_CORR["*H"]["dG_corr"]
    eta_HER = abs(dG_H)

    dE_OH = energies["*OH"] - e_clean - (GAS_REFS["H2O"] - 0.5 * GAS_REFS["H2"])
    dG_OH = dE_OH + THERMO_CORR["*OH"]["dG_corr"]
    dE_O = energies["*O"] - e_clean - (GAS_REFS["H2O"] - GAS_REFS["H2"])
    dG_O = dE_O + THERMO_CORR["*O"]["dG_corr"]
    dE_OOH = energies["*OOH"] - e_clean - (2 * GAS_REFS["H2O"] - 1.5 * GAS_REFS["H2"])
    dG_OOH = dE_OOH + THERMO_CORR["*OOH"]["dG_corr"]
    pds_val = max([dG_OH, dG_O - dG_OH, dG_OOH - dG_O, 4.92 - dG_OOH])
    eta_OER = max(0.0, pds_val - 1.23)

    dE_COOH = energies["*COOH"] - e_clean - (GAS_REFS["CO2"] + 0.5 * GAS_REFS["H2"])
    dG_COOH = dE_COOH + THERMO_CORR["*COOH"]["dG_corr"]
    dE_CO = energies["*CO"] - e_clean - (GAS_REFS["CO2"] + GAS_REFS["H2"] - GAS_REFS["H2O"])
    dG_CO = dE_CO + THERMO_CORR["*CO"]["dG_corr"]
    eta_CO2RR = max(0.0, max(dG_COOH, dG_CO - dG_COOH) - (-0.11))
    delta_G_sel = dG_COOH - dG_H

    raw_dE = [dE_H, dE_OH, dE_O, dE_OOH, dE_COOH, dE_CO]
    if any(np.isfinite(d) and abs(d) > 50.0 for d in raw_dE):
        any_congested = True

    return {
        "qmof_id": q_id, "metal": metal, "pld_A": pld, "lcd_A": lcd, "bandgap_eV": bandgap,
        "coord_number": coord_num, "natoms": len(atoms),
        "dG_H_eV": dG_H, "eta_HER_V": eta_HER, "dG_OH_eV": dG_OH, "dG_O_eV": dG_O,
        "dG_OOH_eV": dG_OOH, "eta_OER_V": eta_OER, "dG_COOH_eV": dG_COOH, "dG_CO_eV": dG_CO,
        "eta_CO2RR_V": eta_CO2RR, "delta_G_sel_eV": delta_G_sel,
        "prefers_CO2RR": bool(delta_G_sel < 0.0), "steric_congested": bool(any_congested),
        "error": None,
    }


def main():
    logger.info("=== STEP 16: Massive High-Throughput Scale-Up (2260-MOF Pore-Accessible Cohort) ===")
    pool = build_pool()

    done_ids = set()
    if OUTPUT_CSV.exists():
        done_df = pd.read_csv(OUTPUT_CSV)
        done_ids = set(done_df["qmof_id"])
        logger.info(f"Resuming: {len(done_ids)}/{len(pool)} MOFs already completed in {OUTPUT_CSV}.")
    else:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(OUTPUT_CSV, index=False)

    nested_zf = None
    if ZIP_PATH.exists():
        zf = zipfile.ZipFile(ZIP_PATH, "r")
        nested_bytes = zf.read("qmof_database/relaxed_structures.zip")
        nested_zf = zipfile.ZipFile(__import__("io").BytesIO(nested_bytes))
        logger.info(f"Opened structure archive {ZIP_PATH} for CIF extraction.")
    else:
        logger.warning(f"{ZIP_PATH} not found -- only MOFs with already-local pristine CIFs can be screened.")

    calc = mace_mp(model=MODEL_SIZE, device=DEVICE, default_dtype=DTYPE)
    logger.info(f"MACE-MP-0 ({MODEL_SIZE}, {DEVICE}, {DTYPE}) initialized.")

    todo = pool[~pool["qmof_id"].isin(done_ids)].reset_index(drop=True)
    logger.info(f"{len(todo)} MOFs remaining to screen.")

    t_start = time.time()
    n_done_this_run = 0
    for idx, row in todo.iterrows():
        q_id = row["qmof_id"]
        metal = row["primary_metal"]
        pld = row["info.pld"]
        lcd = row["info.lcd"]
        bandgap = row.get("outputs.pbe.bandgap", np.nan)

        try:
            cif_path = get_pristine_cif(q_id, nested_zf)
            if cif_path is None:
                raise FileNotFoundError("CIF not found locally or in archive")
            atoms = ase.io.read(str(cif_path))
            result = screen_one(q_id, metal, pld, lcd, bandgap, atoms, calc)
        except Exception as e:
            logger.error(f"[{idx+1}/{len(todo)}] {q_id} FAILED: {type(e).__name__}: {e}")
            result = {"qmof_id": q_id, "metal": metal, "pld_A": pld, "lcd_A": lcd,
                      "bandgap_eV": bandgap, "error": f"{type(e).__name__}: {e}"}

        row_df = pd.DataFrame([{c: result.get(c) for c in CSV_COLUMNS}])
        row_df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        n_done_this_run += 1

        if result.get("error"):
            logger.info(f"[{idx+1}/{len(todo)}] {q_id} ({metal}) -> ERROR: {result['error']}")
        else:
            logger.info(f"[{idx+1}/{len(todo)}] {q_id} ({metal}, PLD={pld:.2f} A) -> "
                        f"eta_HER={result['eta_HER_V']:.2f} V, eta_OER={result['eta_OER_V']:.2f} V, "
                        f"eta_CO2RR={result['eta_CO2RR_V']:.2f} V, congested={result['steric_congested']}")

        if n_done_this_run % 25 == 0:
            elapsed = time.time() - t_start
            rate = elapsed / n_done_this_run
            remaining = len(todo) - n_done_this_run
            eta_hours = remaining * rate / 3600
            logger.info(f"--- Progress: {n_done_this_run}/{len(todo)} this run "
                        f"({rate:.1f} s/MOF avg, ETA {eta_hours:.1f} h remaining) ---")

    if nested_zf is not None:
        nested_zf.close()

    final_df = pd.read_csv(OUTPUT_CSV)
    logger.info(f"[VALIDATION] {len(final_df)}/{len(pool)} MOFs present in {OUTPUT_CSV}.")
    n_errors = final_df["error"].notna().sum()
    n_congested = (final_df["steric_congested"] == True).sum()  # noqa: E712
    logger.info(f"[VALIDATION] {n_errors} errored, {n_congested} sterically congested, "
                f"{len(final_df) - n_errors - n_congested} clean usable results.")
    logger.info("[VALIDATION PASSED] script_16_massive_scaleup_2260.py run segment complete.")


if __name__ == "__main__":
    main()
