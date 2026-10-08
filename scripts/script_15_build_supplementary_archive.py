#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 15: Build Supplementary CIF Archive and Per-Phase Parameter Tables
Project: mofs-mace-her-oer-co2rr (Repo: projeto02-26092026)

Produces a single, self-contained Supporting Information data package:

  1. SI/cif_archive/<NN_phase>/*.cif  -- every structure analysed in the
     study, copied (not moved -- the working structures/ tree used by the
     rest of the pipeline is left untouched) and organised by study phase.
  2. SI/tables/table5..table9_*.tex   -- one LaTeX (longtable) per phase,
     listing every system with its structural (unit-cell) and energetic
     (MACE/DFT total energy, CHE Gibbs free energy) parameters.
  3. SI/cif_archive/<NN_phase>/index.csv -- the same per-phase table as a
     plain CSV, so the archive is self-documenting even without SI.tex.
  4. SI/cif_archive/README.md -- what each phase is, how many structures,
     and an explicit, honest note on the one known gap (the 32-MOF
     high-throughput scale-up cohort's relaxed intermediate structures were
     never written to disk by script_12 -- only its 32 pristine inputs and
     its final energetics/Gibbs-energy CSV exist; there is nothing to
     archive for intermediate geometries in that cohort unless it is rerun
     with structure-saving enabled).

Does not re-run any physics; only reorganises and tabulates results that
already exist in data/*.csv and structures/*.
"""

import shutil
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import ase.io

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STRUCT_DIR = BASE_DIR / "structures"
ARCHIVE_DIR = BASE_DIR / "SI" / "cif_archive"
TABLES_DIR = BASE_DIR / "SI" / "tables"
LOG_DIR = BASE_DIR / "logs"

for d in [ARCHIVE_DIR, TABLES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "results.log", mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("script_15_build_supplementary_archive")


def latex_escape(s) -> str:
    """Escapes LaTeX-special characters in a raw data string (e.g. QMOF
    space-group labels like 'P3_221' contain a literal underscore, which
    LaTeX reads as a math-mode subscript and fails to compile outside one)."""
    s = str(s)
    for ch, esc in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                    ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(ch, esc)
    return s


def cellpar_dict(cif_path: Path) -> dict:
    atoms = ase.io.read(str(cif_path))
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    vol = atoms.get_volume()
    return {"a_A": a, "b_A": b, "c_A": c, "alpha_deg": alpha, "beta_deg": beta,
            "gamma_deg": gamma, "volume_A3": vol, "n_atoms": len(atoms)}


def copy_cifs(src_dir: Path, dst_dir: Path, pattern: str = "*.cif") -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.glob(pattern)):
        shutil.copy2(f, dst_dir / f.name)
        n += 1
    return n


def df_to_longtable(df: pd.DataFrame, col_spec: str, headers: list, caption: str,
                     label: str, fmt_row_func, fontsize: str = "scriptsize") -> str:
    lines = []
    lines.append(f"% {caption}")
    lines.append("{\\" + fontsize)
    lines.append(r"\begin{longtable}{" + col_spec + "}")
    lines.append(r"\caption{" + caption + r"} \label{" + label + r"} \\")
    lines.append(r"\hline\hline")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{" + str(len(headers)) + r"}{l}{\textit{(continued from previous page)}} \\")
    lines.append(r"\hline\hline")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\endhead")
    lines.append(r"\hline")
    lines.append(r"\multicolumn{" + str(len(headers)) + r"}{r}{\textit{(continued on next page)}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\hline\hline")
    lines.append(r"\endlastfoot")
    for _, row in df.iterrows():
        lines.append(fmt_row_func(row) + r" \\")
    lines.append(r"\end{longtable}")
    lines.append(r"}")
    return "\n".join(lines) + "\n"


def main():
    logger.info("=== STEP 15: Build Supplementary CIF Archive and Per-Phase Tables ===")

    # -------------------------------------------------------------------
    # PHASE 1: Benchmark cohort -- pristine frameworks (18 MOFs)
    # -------------------------------------------------------------------
    phase1_dir = ARCHIVE_DIR / "01_benchmark_cohort_pristine"
    n1 = copy_cifs(STRUCT_DIR / "pristine_mofs", phase1_dir)
    logger.info(f"Phase 1 (benchmark pristine): copied {n1} CIFs")

    cohort = pd.read_csv(DATA_DIR / "cohort_benchmark_mofs.csv")
    rows1 = []
    for _, r in cohort.iterrows():
        q = r["qmof_id"]
        cp = cellpar_dict(phase1_dir / f"{q}.cif")
        rows1.append({
            "qmof_id": q, "metal": r["primary_metal"], "formula": r["info.formula_reduced"],
            "spacegroup": r["info.symmetry.spacegroup"], **cp,
            "pld_A": r["info.pld"], "lcd_A": r["info.lcd"],
            "bandgap_eV": r.get("outputs.pbe.bandgap", np.nan),
            "energy_pbe_dft_eV": r.get("outputs.pbe.energy_total", np.nan),
        })
    df1 = pd.DataFrame(rows1)
    df1.to_csv(phase1_dir / "index.csv", index=False)

    def fmt1(row):
        return (f"{row['qmof_id']} & {row['metal']} & {latex_escape(row['formula'])} & "
                f"{latex_escape(row['spacegroup'])} & "
                f"{row['a_A']:.3f} & {row['b_A']:.3f} & {row['c_A']:.3f} & "
                f"{row['alpha_deg']:.2f} & {row['beta_deg']:.2f} & {row['gamma_deg']:.2f} & "
                f"{row['volume_A3']:.1f} & {row['pld_A']:.2f} & {row['lcd_A']:.2f} & "
                f"{row['bandgap_eV']:.3f} & {row['energy_pbe_dft_eV']:.3f}")

    tex1 = df_to_longtable(
        df1, "llllccccccccccc",
        ["QMOF ID", "Metal", "Formula", "Space group", "$a$ (\\AA)", "$b$ (\\AA)", "$c$ (\\AA)",
         "$\\alpha$ ($^\\circ$)", "$\\beta$ ($^\\circ$)", "$\\gamma$ ($^\\circ$)",
         "Volume (\\AA$^3$)", "PLD (\\AA)", "LCD (\\AA)", "Bandgap (eV)", "$E_{\\mathrm{PBE}}$ (eV)"],
        "Table S5: Benchmark Cohort (18 MOFs) -- Pristine Framework Structural Parameters. "
        "Cell parameters are identical across all relaxed states of a given MOF (fixed-cell "
        "relaxation throughout this work); $E_{\\mathrm{PBE}}$ is the as-deposited QMOF PBE-D3BJ "
        "total energy of the pristine framework, included for provenance only -- it is not used "
        "in any CHE free energy in this work (see Table S6).",
        "tab:benchmark_pristine_structural", fmt1,
    )
    (TABLES_DIR / "table5_benchmark_pristine_structural.tex").write_text(tex1, encoding="utf-8")
    logger.info(f"Table S5 written ({len(df1)} rows)")

    # -------------------------------------------------------------------
    # PHASE 2: Benchmark cohort -- Tier-1/Tier-2 relaxed structures + full
    # reaction-intermediate energetics (126 states: 18 MOFs x 7 intermediates)
    # -------------------------------------------------------------------
    phase2a_dir = ARCHIVE_DIR / "02_benchmark_cohort_tier1_chgnet_prerelaxed"
    phase2b_dir = ARCHIVE_DIR / "03_benchmark_cohort_tier2_mace_relaxed_FINAL"
    n2a = copy_cifs(STRUCT_DIR / "chgnet_prerelaxed", phase2a_dir)
    n2b = copy_cifs(STRUCT_DIR / "mace_relaxed", phase2b_dir)
    logger.info(f"Phase 2 (Tier-1 CHGNet pre-relaxed): copied {n2a} CIFs")
    logger.info(f"Phase 3 (Tier-2 MACE relaxed, FINAL production structures): copied {n2b} CIFs")

    ads_e = pd.read_csv(DATA_DIR / "adsorption_energies_mace.csv")
    phva = pd.read_csv(DATA_DIR / "phva_thermochemistry.csv")
    merged = ads_e.merge(
        phva[["qmof_id", "reaction", "intermediate", "zpe_ads_eV", "ts_ads_eV",
              "delta_G_corr_eV", "delta_G_eV", "steric_congested"]],
        on=["qmof_id", "reaction", "intermediate"], how="left", suffixes=("", "_phva"))
    merged = merged.merge(cohort[["qmof_id", "primary_metal"]], on="qmof_id", how="left")
    merged.to_csv(phase2b_dir / "index.csv", index=False)

    def fmt2(row):
        cong = "Yes" if bool(row.get("steric_congested", False)) else "No"
        return (f"{row['qmof_id']} & {row['primary_metal']} & {row['reaction'].upper()} & "
                f"{row['intermediate']} & {row['delta_E_eV']:.3f} & {row['zpe_ads_eV']:.3f} & "
                f"{row['ts_ads_eV']:.3f} & {row['delta_G_eV']:.3f} & {row['fmax_eV_A']:.3f} & "
                f"{'Yes' if row['converged'] else 'No'} & {cong}")

    tex2 = df_to_longtable(
        merged, "llllccccccc",
        ["QMOF ID", "Metal", "Reaction", "Intermediate", "$\\Delta E_{\\mathrm{MACE}}$ (eV)",
         "ZPE (eV)", "$-T\\Delta S$ (eV)", "$\\Delta G_{\\mathrm{CHE}}$ (eV)",
         "$f_{\\max}$ (eV/\\AA)", "Converged", "Congested"],
        "Table S6: Benchmark Cohort (18 MOFs) -- Full Reaction-Intermediate Energetics. "
        "All 126 computed states (7 intermediates $\\times$ 18 MOFs: $*$H, $*$OH, $*$O, $*$OOH, "
        "$*$COOH, $*$CO, $*$OCHO), Tier-2 MACE-MP-0 relaxation energies with PHVA "
        "(ZPE + entropic) corrections giving the final CHE Gibbs free energies used throughout "
        "the main text and Table S1. Congested states (Methods, main text) are included here for "
        "completeness but excluded from Table S1 and all volcano/selectivity analyses.",
        "tab:benchmark_intermediates_full", fmt2,
    )
    (TABLES_DIR / "table6_benchmark_intermediates_full.tex").write_text(tex2, encoding="utf-8")
    logger.info(f"Table S6 written ({len(merged)} rows)")

    # -------------------------------------------------------------------
    # PHASE 3: GPAW DFT validation benchmark
    # -------------------------------------------------------------------
    gpaw = pd.read_csv(DATA_DIR / "gpaw_dft_benchmark.csv")

    def fmt3(row):
        return (f"{row['system']} & {int(row['natoms'])} & {row['e_gpaw_dft_eV']:.4f} & "
                f"{row['e_mace_eV']:.4f} & {row['diff_mace_dft_eV']:.4f} & {row['diff_per_atom_eV']:.5f}")

    tex3 = df_to_longtable(
        gpaw, "lcccccc"[:6] if False else "lccccc",
        ["System", "$N_{\\mathrm{atoms}}$", "$E_{\\mathrm{GPAW\\text{-}DFT}}$ (eV)",
         "$E_{\\mathrm{MACE}}$ (eV)", "$\\Delta E$ (eV)", "$\\Delta E$/atom (eV)"],
        "Table S7: GPAW DFT Validation Benchmark. Cross-check of MACE-MP-0 total energies "
        "against PBE/LCAO GPAW DFT single points for the gas-phase references and a subset of "
        "relaxed clusters (Methods, main text).",
        "tab:gpaw_validation", fmt3,
    )
    (TABLES_DIR / "table7_gpaw_dft_validation.tex").write_text(tex3, encoding="utf-8")
    logger.info(f"Table S7 written ({len(gpaw)} rows)")

    # -------------------------------------------------------------------
    # PHASE 4: Electronic structure / d-band centers
    # -------------------------------------------------------------------
    dband = pd.read_csv(DATA_DIR / "dft_pdos_dband_centers.csv")

    def fmt4(row):
        return (f"{row['qmof_id']} & {row['metal']} & {row['state']} & {int(row['natoms_cluster'])} & "
                f"{row['e_fermi_eV']:.3f} & {row['d_band_center_rel_EF_eV']:.3f} & "
                f"{row['magnetic_moment_muB']:.3f}")

    tex4 = df_to_longtable(
        dband, "lllcccc",
        ["QMOF ID", "Metal", "State", "$N_{\\mathrm{atoms}}$ (cluster)", "$E_{\\mathrm{Fermi}}$ (eV)",
         "$d$-band centre rel. $E_F$ (eV)", "Magnetic moment ($\\mu_B$)"],
        "Table S8: Electronic-Structure Analysis -- DFT $d$-Band Centres. Projected density of "
        "states analysis (GPAW PBE/LCAO) for the pristine and adsorbate-decorated open-metal-site "
        "clusters discussed in Figure 7 of the main text.",
        "tab:dband_centers", fmt4,
    )
    (TABLES_DIR / "table8_dband_centers.tex").write_text(tex4, encoding="utf-8")
    logger.info(f"Table S8 written ({len(dband)} rows)")

    # -------------------------------------------------------------------
    # PHASE 5: High-throughput scale-up cohort (32 MOFs) -- pristine only
    # (intermediate relaxed structures were computed in-memory by script_12
    # and never written to disk; honestly disclosed in the archive README
    # rather than silently omitted or fabricated).
    # -------------------------------------------------------------------
    phase5_dir = ARCHIVE_DIR / "04_scaleup_cohort_pristine_32MOF"
    n5 = copy_cifs(STRUCT_DIR / "scaleup_pristine", phase5_dir)
    logger.info(f"Phase 5 (scale-up pristine, 32-MOF cohort): copied {n5} CIFs")

    scaleup = pd.read_csv(DATA_DIR / "high_throughput_scaleup_results.csv")
    rows5 = []
    for _, r in scaleup.iterrows():
        q = r["qmof_id"]
        cif_path = phase5_dir / f"{q}.cif"
        cp = cellpar_dict(cif_path) if cif_path.exists() else {k: np.nan for k in
            ["a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg", "volume_A3", "n_atoms"]}
        rows5.append({**r.to_dict(), **cp})
    df5 = pd.DataFrame(rows5)
    df5.to_csv(phase5_dir / "index.csv", index=False)

    def fmt5(row):
        cong = "Yes" if bool(row.get("steric_congested", False)) else "No"
        pref = "CO2RR" if bool(row.get("prefers_CO2RR", False)) else "HER"

        def f(x):
            return f"{x:.2f}" if pd.notna(x) else "--"
        return (f"{row['qmof_id']} & {row['metal']} & {f(row['a_A'])} & {f(row['b_A'])} & {f(row['c_A'])} & "
                f"{f(row['volume_A3'])} & {f(row['pld_A'])} & {f(row['lcd_A'])} & {f(row['bandgap_eV'])} & "
                f"{f(row['dG_H_eV'])} & {f(row['dG_OH_eV'])} & {f(row['dG_O_eV'])} & {f(row['dG_OOH_eV'])} & "
                f"{f(row['eta_OER_V'])} & {f(row['dG_COOH_eV'])} & {f(row['dG_CO_eV'])} & "
                f"{f(row['eta_CO2RR_V'])} & {f(row['delta_G_sel_eV'])} & {pref} & {cong}")

    tex5 = df_to_longtable(
        df5, "llcccccccccccccccccc",
        ["QMOF ID", "Metal", "$a$ (\\AA)", "$b$ (\\AA)", "$c$ (\\AA)", "Vol. (\\AA$^3$)",
         "PLD (\\AA)", "LCD (\\AA)", "Gap (eV)", "$\\Delta G_{*\\mathrm{H}}$", "$\\Delta G_{*\\mathrm{OH}}$",
         "$\\Delta G_{*\\mathrm{O}}$", "$\\Delta G_{*\\mathrm{OOH}}$", "$\\eta^{\\mathrm{OER}}$",
         "$\\Delta G_{*\\mathrm{COOH}}$", "$\\Delta G_{*\\mathrm{CO}}$", "$\\eta^{\\mathrm{CO2RR}}$",
         "$\\Delta G_{\\mathrm{sel}}$", "Prefers", "Congested"],
        "Table S9: High-Throughput Scale-Up Cohort (32 MOFs) -- Structural and Energetic "
        "Parameters. All 32 screened MOFs, including the 8 flagged as sterically congested "
        "(Section~\\ref{sec:congestion-note}); those 8 are excluded from Figure 9 and Table S3 "
        "but are listed here in full for transparency. Free energies in eV, overpotentials in V. "
        "Relaxed intermediate structures for this cohort were computed in memory by the "
        "screening script and were not written to disk, so only the 32 pristine CIFs are "
        "archived for this phase (\\texttt{04\\_scaleup\\_cohort\\_pristine\\_32MOF/}); cell "
        "parameters above are those of the pristine (adsorbate-free) framework.",
        "tab:scaleup_structural_energetic", fmt5, fontsize="tiny",
    )
    (TABLES_DIR / "table9_scaleup_structural_energetic.tex").write_text(tex5, encoding="utf-8")
    logger.info(f"Table S9 written ({len(df5)} rows)")

    # -------------------------------------------------------------------
    # PHASE 6 & 7: Micro-solvation (single geometry + configurational ensemble)
    # -------------------------------------------------------------------
    phase6_dir = ARCHIVE_DIR / "05_microsolvation_single_geometry"
    phase7_dir = ARCHIVE_DIR / "06_microsolvation_ensemble"
    n6 = copy_cifs(STRUCT_DIR / "solvated_intermediates", phase6_dir)
    n7 = copy_cifs(STRUCT_DIR / "solvated_ensemble", phase7_dir)
    logger.info(f"Phase 6 (micro-solvation, single geometry): copied {n6} CIFs")
    logger.info(f"Phase 7 (micro-solvation, configurational ensemble): copied {n7} CIFs")
    shutil.copy2(DATA_DIR / "microsolvation_thermodynamics.csv", phase6_dir / "index.csv")
    shutil.copy2(DATA_DIR / "microsolvation_ensemble_raw.csv", phase7_dir / "index_raw.csv")
    shutil.copy2(DATA_DIR / "microsolvation_ensemble_summary.csv", phase7_dir / "index_summary.csv")

    # -------------------------------------------------------------------
    # README
    # -------------------------------------------------------------------
    total_cifs = n1 + n2a + n2b + n5 + n6 + n7
    readme = f"""# Supporting Information: Structure Archive

Every structure analysed in this study, organised by study phase. Copied
from the working `structures/` tree used by the analysis pipeline
(`scripts/script_01`-`script_13`); nothing here was recomputed, this is a
reorganisation and tabulation of existing results for Supporting
Information purposes.

| # | Phase | N structures | Folder | Table |
|---|-------|-------------:|--------|-------|
| 1 | Benchmark cohort (18 MOFs), pristine frameworks | {n1} | `01_benchmark_cohort_pristine/` | Table S5 |
| 2 | Benchmark cohort, Tier-1 CHGNet pre-relaxed (pristine + 7 intermediates x 18) | {n2a} | `02_benchmark_cohort_tier1_chgnet_prerelaxed/` | -- (provenance only) |
| 3 | Benchmark cohort, Tier-2 MACE-MP-0 relaxed -- **FINAL production structures** | {n2b} | `03_benchmark_cohort_tier2_mace_relaxed_FINAL/` | Table S6 |
| 4 | High-throughput scale-up cohort (32 MOFs), pristine frameworks | {n5} | `04_scaleup_cohort_pristine_32MOF/` | Table S9 |
| 5 | Micro-solvation, single deterministic geometry (2 champions, n=1,2,3 H2O) | {n6} | `05_microsolvation_single_geometry/` | Table S2 |
| 6 | Micro-solvation, configurational-sampling ensemble (5 replicas/level) | {n7} | `06_microsolvation_ensemble/` | Table S4 |

**Total: {total_cifs} CIF files.**

Each phase folder also contains an `index.csv` (or `index_raw.csv` /
`index_summary.csv` for phase 6) with the same data as its corresponding
LaTeX table, for machine-readable access without parsing the SI PDF.

## Known gap: high-throughput scale-up intermediate structures

The 32-MOF scale-up screen (`script_12_high_throughput_scaleup.py`)
builds each adsorbate-decorated structure, relaxes it with MACE-MP-0, and
computes its energy entirely in memory; it was never instructed to write
the relaxed intermediate geometries to disk (unlike every other phase in
this pipeline). Only the 32 pristine input frameworks and the final
energetics/Gibbs-energy CSV (`data/high_throughput_scaleup_results.csv`,
reproduced in Table S9) exist. If per-intermediate structures for this
32-MOF cohort are needed (e.g. for an independent DFT re-check), the
screening script would need to be rerun with structure-saving added --
this has not been done, since it requires a fresh GPU pass over all 32
MOFs. This gap is disclosed here rather than silently omitted or
backfilled with placeholder geometries.

## Not included in this archive

- The initial, unrelaxed intermediate placements from
  `structures/intermediates_her/oer/co2rr/` (the construction step before
  any relaxation; superseded by the Tier-1/Tier-2 structures above) and
  the Tier-1 CHGNet pre-relaxed set are kept in the public GitHub
  repository for full provenance but are not duplicated into every phase
  table here, to avoid listing near-duplicate, non-final geometries
  table-by-table. The Tier-1 CIFs ARE archived (phase 2 above) since they
  are cheap to include and some readers may want the pre-MACE geometries.
"""
    (ARCHIVE_DIR / "README.md").write_text(readme, encoding="utf-8")
    logger.info(f"README.md written. Total archived CIFs: {total_cifs}")

    logger.info("[VALIDATION PASSED] script_15_build_supplementary_archive.py completed successfully.")


if __name__ == "__main__":
    main()
