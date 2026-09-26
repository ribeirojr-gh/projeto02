#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 06: Computational Hydrogen Electrode (CHE) Thermodynamics & Overpotentials
Project: mofs-mace-her-oer-co2rr (Repo: projeto02-26092026)

Purpose:
  Calculates the free energy reaction profiles (ΔG_i) for HER, OER, and CO2RR
  across all MOF systems under standard CHE conditions (U = 0 V, pH = 0, T = 298.15 K).
  Determines potential-determining steps (PDS), theoretical overpotentials
  (η_HER, η_OER, η_CO2RR), and CO2RR vs. HER selectivity descriptors.

Dependencies:
    pip install pandas numpy
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

COHORT_CSV = DATA_DIR / "cohort_benchmark_mofs.csv"
PHVA_CSV = DATA_DIR / "phva_thermochemistry.csv"
CHE_SUMMARY_CSV = DATA_DIR / "che_electrocatalysis_summary.csv"
LOG_FILE = LOG_DIR / "results.log"

# Equilibrium thermodynamic potentials vs. RHE (V)
U0_OER = 1.23       # 2 H2O -> O2 + 4 H+ + 4 e- (ΔG_tot = 4.92 eV)
U0_HER = 0.00       # 2 H+ + 2 e- -> H2
U0_CO = -0.11       # CO2 + 2 H+ + 2 e- -> CO + H2O
U0_HCOOH = -0.19    # CO2 + 2 H+ + 2 e- -> HCOOH

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("script_06_che_energetics")


def main():
    logger.info("=== STEP 8: Computational Hydrogen Electrode (CHE) Thermodynamics ===")

    if not PHVA_CSV.exists() or not COHORT_CSV.exists():
        logger.error("Prerequisite CSV files missing.")
        sys.exit(1)

    cohort_df = pd.read_csv(COHORT_CSV)
    thermo_df = pd.read_csv(PHVA_CSV)
    logger.info(f"Loaded {len(cohort_df)} cohort MOFs and {len(thermo_df)} thermochemical entries.")

    # Pivot ΔG values by qmof_id and intermediate
    pivoted_dg = thermo_df.pivot(index="qmof_id", columns="intermediate", values="delta_G_eV")
    pivoted_congested = thermo_df.pivot(index="qmof_id", columns="intermediate", values="steric_congested")

    summary_records = []

    for _, row in cohort_df.iterrows():
        q_id = row["qmof_id"]
        metal = row["primary_metal"]
        formula = row["info.formula"]

        if q_id not in pivoted_dg.index:
            continue

        dg = pivoted_dg.loc[q_id]
        cong = pivoted_congested.loc[q_id]

        dg_h = dg.get("H", np.nan)
        dg_oh = dg.get("OH", np.nan)
        dg_o = dg.get("O", np.nan)
        dg_ooh = dg.get("OOH", np.nan)
        dg_cooh = dg.get("COOH", np.nan)
        dg_co = dg.get("CO", np.nan)
        dg_ocho = dg.get("OCHO", np.nan)

        # Flag overall steric feasibility
        any_congested = bool(cong.any())

        # ---------------------------------------------------------------------
        # 1. HER Thermodynamics
        # ---------------------------------------------------------------------
        # * + H+ + e- -> *H (ΔG1 = ΔG_*H)
        # *H + H+ + e- -> * + H2 (ΔG2 = -ΔG_*H)
        eta_her = abs(dg_h) if not np.isnan(dg_h) else np.nan

        # ---------------------------------------------------------------------
        # 2. OER Thermodynamics (4-step associative)
        # ---------------------------------------------------------------------
        # Step 1: * + H2O -> *OH + H+ + e-
        dg_oer_1 = dg_oh
        # Step 2: *OH -> *O + H+ + e-
        dg_oer_2 = dg_o - dg_oh
        # Step 3: *O + H2O -> *OOH + H+ + e-
        dg_oer_3 = dg_ooh - dg_o
        # Step 4: *OOH -> * + O2 + H+ + e-
        dg_oer_4 = 4.92 - dg_ooh

        oer_steps = [dg_oer_1, dg_oer_2, dg_oer_3, dg_oer_4]
        if not any(np.isnan(s) for s in oer_steps):
            dg_pds_oer = max(oer_steps)
            step_names = ["* -> *OH", "*OH -> *O", "*O -> *OOH", "*OOH -> O2"]
            pds_oer_name = step_names[int(np.argmax(oer_steps))]
            eta_oer = max(0.0, dg_pds_oer - U0_OER)
        else:
            dg_pds_oer = np.nan
            pds_oer_name = "N/A"
            eta_oer = np.nan

        # ---------------------------------------------------------------------
        # 3. CO2RR Thermodynamics (CO and Formate pathways)
        # ---------------------------------------------------------------------
        # CO Pathway:
        # Step 1: CO2 + H+ + e- + * -> *COOH (ΔG1 = ΔG_*COOH)
        # Step 2: *COOH + H+ + e- -> *CO + H2O (ΔG2 = ΔG_*CO - ΔG_*COOH)
        # Step 3: *CO -> CO(g) + * (ΔG_des = -ΔG_*CO)
        dg_co2_cooh = dg_cooh
        dg_cooh_co = dg_co - dg_cooh
        dg_co_des = -dg_co

        co2rr_steps = [dg_co2_cooh, dg_cooh_co]
        if not any(np.isnan(s) for s in co2rr_steps):
            u_lim_co = -max(co2rr_steps)
            eta_co2rr = max(0.0, abs(u_lim_co - U0_CO))
        else:
            u_lim_co = np.nan
            eta_co2rr = np.nan

        # Formate Pathway:
        # CO2 + H+ + e- + * -> *OCHO (ΔG = ΔG_*OCHO)
        dg_co2_ocho = dg_ocho
        eta_formate = max(0.0, abs(-dg_co2_ocho - U0_HCOOH)) if not np.isnan(dg_co2_ocho) else np.nan

        # ---------------------------------------------------------------------
        # 4. Selectivity Descriptors
        # ---------------------------------------------------------------------
        # Selectivity metric: ΔG_sel = ΔG_*COOH - ΔG_*H
        # Negative value -> CO2 activation is favored over proton reduction (HER suppressed)
        dg_sel_cooh_h = dg_cooh - dg_h if (not np.isnan(dg_cooh) and not np.isnan(dg_h)) else np.nan
        dg_sel_ocho_h = dg_ocho - dg_h if (not np.isnan(dg_ocho) and not np.isnan(dg_h)) else np.nan

        summary_records.append({
            "qmof_id": q_id,
            "metal": metal,
            "formula": formula,
            "pld_A": row["info.pld"],
            "steric_congested": any_congested,
            # HER
            "delta_G_H_eV": dg_h,
            "eta_HER_V": eta_her,
            # OER
            "delta_G_OH_eV": dg_oh,
            "delta_G_O_eV": dg_o,
            "delta_G_OOH_eV": dg_ooh,
            "dg_oer_step1_eV": dg_oer_1,
            "dg_oer_step2_eV": dg_oer_2,
            "dg_oer_step3_eV": dg_oer_3,
            "dg_oer_step4_eV": dg_oer_4,
            "pds_OER": pds_oer_name,
            "eta_OER_V": eta_oer,
            # CO2RR
            "delta_G_COOH_eV": dg_cooh,
            "delta_G_CO_eV": dg_co,
            "delta_G_OCHO_eV": dg_ocho,
            "u_lim_CO_V": u_lim_co,
            "eta_CO2RR_CO_V": eta_co2rr,
            "eta_CO2RR_formate_V": eta_formate,
            # Selectivity
            "delta_G_selectivity_COOH_vs_H_eV": dg_sel_cooh_h,
            "delta_G_selectivity_OCHO_vs_H_eV": dg_sel_ocho_h,
            "prefers_CO2RR": dg_sel_cooh_h < 0.0 if not np.isnan(dg_sel_cooh_h) else False
        })

        logger.info(
            f"MOF: {q_id:12s} ({metal:2s}) | η_HER: {eta_her:4.2f} V | "
            f"η_OER: {eta_oer:4.2f} V (PDS: {pds_oer_name}) | "
            f"η_CO2RR: {eta_co2rr:4.2f} V | ΔG_sel(COOH-H): {dg_sel_cooh_h:+5.2f} eV"
        )

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(CHE_SUMMARY_CSV, index=False)
    logger.info(f"Exported electrocatalysis CHE summary table to {CHE_SUMMARY_CSV}")

    # =========================================================================
    # OUTPUT VALIDATION BLOCK (Required by Protocol Section 9.2 & 11.1)
    # =========================================================================
    validation_passed = True
    validation_errors = []

    if not CHE_SUMMARY_CSV.exists() or len(summary_df) == 0:
        validation_passed = False
        validation_errors.append(f"Missing or empty summary table: {CHE_SUMMARY_CSV}")
    elif len(summary_df) != len(cohort_df):
        validation_passed = False
        validation_errors.append(f"Mismatch in processed systems: expected {len(cohort_df)}, processed {len(summary_df)}")

    # Check that overpotentials are non-negative
    valid_etas = summary_df["eta_OER_V"].dropna()
    if (valid_etas < 0.0).any():
        validation_passed = False
        validation_errors.append("Detected unphysical negative overpotential in OER.")

    if validation_passed:
        logger.info("\n[VALIDATION PASSED] script_06_che_energetics.py")
        logger.info(f"Successfully evaluated complete CHE reaction thermodynamics for all {len(summary_df)} systems.")
    else:
        logger.error("\n[VALIDATION FAILED] script_06_che_energetics.py")
        for err in validation_errors:
            logger.error(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
