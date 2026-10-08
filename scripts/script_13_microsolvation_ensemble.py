#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 13: Micro-Solvation Configurational-Sampling Ensemble
Project: mofs-mace-her-oer-co2rr (Repo: projeto02-26092026)

Complements script_11 (single deterministic water placement per hydration
level) by sampling multiple independent, randomly-rotated water placements
per hydration level (n = 1, 2, 3 H2O) for the two champion MOFs, so the
micro-solvation result is backed by a small ensemble rather than one
geometry. Does NOT modify script_11's outputs (Fig. 8 / Table S2) -- this
produces separate, additional outputs.

Convergence policy (relaxed vs. script_11, by explicit user request):
  Attempt fmax targets in increasing (looser) tiers, continuing the SAME
  FIRE trajectory across tiers (no restart): 0.03 -> 0.05 -> 0.08 -> 0.12
  eV/A, each with a further 250-step budget. Every row records which tier
  was actually met (fmax_tier_eV_A) and the final residual force
  (fmax_final_eV_A), so a looser-tier result is never silently reported as
  having met the project's standard 0.03 eV/A criterion. Rows that fail to
  converge at even the loosest tier, or that still land on an unphysical
  gas-referenced energy (|delta_E_solv| > 6 eV, same convention as
  script_11/script_05), are flagged and excluded from the aggregated
  summary and figure -- only disclosed in the raw per-replica CSV.

Dependencies:
  mace-torch ase pandas numpy matplotlib torch
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ase.io
from ase import Atoms
from ase.optimize import BFGS, FIRE
from mace.calculators import mace_mp

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IN_DIR = BASE_DIR / "structures" / "mace_relaxed"
OUT_DIR = BASE_DIR / "structures" / "solvated_ensemble"
FIG_DIR = BASE_DIR / "figures"
SI_DIR = BASE_DIR / "SI" / "tables"
LOG_DIR = BASE_DIR / "logs"

for d in [OUT_DIR, FIG_DIR, SI_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "microsolvation_ensemble_raw.csv"
SUMMARY_CSV = DATA_DIR / "microsolvation_ensemble_summary.csv"
TEX_OUTPUT = SI_DIR / "table4_microsolvation_ensemble.tex"
LOG_FILE = LOG_DIR / "results.log"

MODEL_SIZE = "medium"
DTYPE = "float64"
DEVICE = "cuda"

N_REPLICAS = 5
FMAX_TIERS = [0.03, 0.05, 0.08, 0.12]   # eV/A, loosened progressively
STEPS_PER_TIER = 250                     # cumulative budget added per tier
UNPHYSICAL_DE_THRESHOLD = 6.0            # eV, gas-referenced (script_11/05 convention)
MAX_ROTATION_DEG = 55.0                  # per-replica random cone around the pore axis

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{amsmath}\usepackage{amssymb}",
    "font.size": 10,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("script_13_microsolvation_ensemble")


def build_h2o(o_pos, h1_vec, h2_vec):
    d_oh = 0.96
    h1_u = h1_vec / np.linalg.norm(h1_vec)
    h2_u = h2_vec / np.linalg.norm(h2_vec)
    return Atoms("H2O", positions=[o_pos, o_pos + d_oh * h1_u, o_pos + d_oh * h2_u])


def random_rotate_vector(v, rng, max_angle_deg):
    v = v / np.linalg.norm(v)
    axis = rng.normal(size=3)
    axis = axis - axis.dot(v) * v
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg))
    v_rot = (v * np.cos(angle) + np.cross(axis, v) * np.sin(angle)
             + axis * np.dot(axis, v) * (1 - np.cos(angle)))
    return v_rot / np.linalg.norm(v_rot)


def find_safe_water_pos(atoms, center, pref_dir, r_range=(2.5, 2.7, 2.9, 3.2)):
    pref_u = pref_dir / np.linalg.norm(pref_dir)
    z_u = pref_u
    aux = np.array([1, 0, 0]) if abs(z_u[0]) < 0.8 else np.array([0, 1, 0])
    x_u = np.cross(aux, z_u)
    x_u /= np.linalg.norm(x_u)
    y_u = np.cross(z_u, x_u)

    best_pos, best_dist = None, 0.0
    for r in r_range:
        for theta in np.linspace(0.05, np.pi / 2.2, 8):
            for phi in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                v = r * (np.sin(theta) * np.cos(phi) * x_u + np.sin(theta) * np.sin(phi) * y_u + np.cos(theta) * z_u)
                cand = center + v
                dists = [np.linalg.norm(atoms.positions[i] - cand) for i in range(len(atoms))]
                min_d = np.min(dists)
                if min_d > best_dist:
                    best_dist = min_d
                    best_pos = cand
                if min_d >= 2.15:
                    return cand
    if best_pos is None:
        best_pos = center + 2.8 * pref_u
    return best_pos


def add_solvation_waters_random(atoms, adsorbate_type, n_waters, pore_normal, rng):
    """Same collision-free placement as script_11, but with a random per-replica
    rotation of the approach direction and random tilt magnitudes for the 2nd/3rd
    water, so repeated calls with different rng produce distinct configurations."""
    solvated = atoms.copy()
    if n_waters == 0:
        return solvated

    pore_u = random_rotate_vector(pore_normal, rng, MAX_ROTATION_DEG)

    ads_lens = {"*H": 1, "*OH": 2, "*O": 1, "*OOH": 3, "*COOH": 4, "*CO": 2, "pristine": 0}
    n_ads = ads_lens.get(adsorbate_type, 1)
    if adsorbate_type == "pristine" or n_ads == 0:
        center = atoms.positions[0]
    else:
        center = np.mean(atoms.positions[-n_ads:], axis=0)

    for w_idx in range(n_waters):
        w_center = center if w_idx == 0 else solvated.positions[-3]
        tilt_mag = rng.uniform(0.2, 0.6) if w_idx > 0 else 0.0
        tilt_axis = rng.normal(size=3)
        tilt_axis /= np.linalg.norm(tilt_axis) + 1e-12
        curr_dir = pore_u + tilt_mag * tilt_axis
        curr_dir /= np.linalg.norm(curr_dir)

        w_o = find_safe_water_pos(solvated, w_center, curr_dir)

        h1_vec = -(w_o - center)
        if np.linalg.norm(h1_vec) < 0.1:
            h1_vec = np.array([0.6, 0.6, 0.3])
        perp = np.cross(h1_vec, pore_u)
        if np.linalg.norm(perp) < 0.1:
            perp = np.array([0.0, 1.0, 0.0])
        h2_vec = 0.5 * h1_vec + 0.8 * perp

        h2o = build_h2o(w_o, h1_vec, h2_vec)
        solvated.extend(h2o)

    return solvated


def relax_staged(atoms, calc):
    """Continue the SAME FIRE trajectory across progressively looser fmax tiers.
    ASE's Dynamics.irun() sets self.max_steps = self.nsteps + steps -- i.e. the
    `steps` argument is an INCREMENT from the current step count, not an
    absolute budget. Passing STEPS_PER_TIER (not a running cumulative total)
    each call therefore correctly adds STEPS_PER_TIER more steps per tier, for
    a true total cap of len(FMAX_TIERS) * STEPS_PER_TIER across all tiers.
    Returns (tier_met_or_None, fmax_final, nsteps_used)."""
    atoms.calc = calc
    dyn = FIRE(atoms, logfile=None)
    tier_met = None
    for tier_fmax in FMAX_TIERS:
        converged = dyn.run(fmax=tier_fmax, steps=STEPS_PER_TIER)
        if converged:
            tier_met = tier_fmax
            break
    fmax_final = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
    return tier_met, fmax_final, dyn.nsteps


def main():
    logger.info("=== STEP 13: Micro-Solvation Configurational-Sampling Ensemble ===")
    logger.info(f"N_REPLICAS={N_REPLICAS}, FMAX_TIERS={FMAX_TIERS}, STEPS_PER_TIER={STEPS_PER_TIER}")

    calc = mace_mp(model=MODEL_SIZE, device=DEVICE, default_dtype=DTYPE)

    h2o_isolated = Atoms("H2O", positions=[[0, 0, 0], [0, 0.757, 0.586], [0, -0.757, 0.586]])
    h2o_isolated.set_cell([18.0, 18.0, 18.0])
    h2o_isolated.center()
    h2o_isolated.calc = calc
    BFGS(h2o_isolated, logfile=None).run(fmax=0.01, steps=25)
    e_h2o_ref = float(h2o_isolated.get_potential_energy())
    logger.info(f"MACE-MP-0 isolated H2O ground state energy: {e_h2o_ref:.5f} eV")

    study_cases = [
        ("qmof-73ded45", "Co", "OER", "*OH", "qmof-73ded45_oer_OH.cif"),
        ("qmof-73ded45", "Co", "OER", "*O", "qmof-73ded45_oer_O.cif"),
        ("qmof-73ded45", "Co", "OER", "*OOH", "qmof-73ded45_oer_OOH.cif"),
        ("qmof-73ded45", "Co", "CO2RR", "*COOH", "qmof-73ded45_co2rr_COOH.cif"),
        ("qmof-73ded45", "Co", "CO2RR", "*CO", "qmof-73ded45_co2rr_CO.cif"),
        ("qmof-73ded45", "Co", "HER", "*H", "qmof-73ded45_her_H.cif"),
        ("qmof-b46c098", "Cu", "HER", "*H", "qmof-b46c098_her_H.cif"),
        ("qmof-b46c098", "Cu", "CO2RR", "*COOH", "qmof-b46c098_co2rr_COOH.cif"),
        ("qmof-b46c098", "Cu", "CO2RR", "*CO", "qmof-b46c098_co2rr_CO.cif"),
    ]

    records = []
    for q_id, metal, rxn, state, cif_name in study_cases:
        cif_path = IN_DIR / cif_name
        if not cif_path.exists():
            logger.warning(f"File {cif_path} not found, skipping {q_id}/{state}.")
            continue

        base_atoms = ase.io.read(str(cif_path))
        dists = base_atoms.get_distances(0, range(1, min(len(base_atoms), 7)), mic=True, vector=True)
        pore_vector = -np.mean(dists, axis=0)
        pore_vector = pore_vector / np.linalg.norm(pore_vector)

        base_atoms.calc = calc
        e_dry = float(base_atoms.get_potential_energy())

        for n_w in [1, 2, 3]:
            for rep in range(N_REPLICAS):
                seed = 1000 * rep + 17 * n_w + sum(ord(c) for c in q_id + state)
                rng = np.random.default_rng(seed)
                tag = f"{q_id}_{state.replace('*', '')}_solv{n_w}w_rep{rep}"
                logger.info(f"Relaxing {tag} (seed={seed})...")

                solv_atoms = add_solvation_waters_random(base_atoms, state, n_w, pore_vector, rng)
                tier_met, fmax_final, nsteps = relax_staged(solv_atoms, calc)

                e_solv = float(solv_atoms.get_potential_energy())
                delta_e_solv = e_solv - e_dry - n_w * e_h2o_ref
                e_hb_per_w = delta_e_solv / n_w

                converged = tier_met is not None
                unphysical = abs(delta_e_solv) > UNPHYSICAL_DE_THRESHOLD

                if not converged:
                    logger.warning(f"  -> {tag}: did not converge at any tier up to {FMAX_TIERS[-1]} eV/A "
                                    f"(fmax_final={fmax_final:.3f}, nsteps={nsteps}).")
                elif unphysical:
                    logger.warning(f"  -> {tag}: converged at tier {tier_met} eV/A but UNPHYSICAL "
                                    f"(delta_E_solv={delta_e_solv:.3f} eV). Flagged, excluded from summary.")
                else:
                    logger.info(f"  -> {tag}: tier={tier_met} eV/A, fmax_final={fmax_final:.4f}, "
                                f"delta_E_solv={delta_e_solv:.3f} eV ({e_hb_per_w:.3f} eV/H2O)")

                out_cif = OUT_DIR / f"{tag}.cif"
                ase.io.write(str(out_cif), solv_atoms, format="cif")

                records.append({
                    "qmof_id": q_id, "metal": metal, "reaction": rxn, "state": state,
                    "n_waters": n_w, "replica": rep, "seed": seed,
                    "total_energy_eV": e_solv, "solvation_energy_eV": delta_e_solv,
                    "h_bond_stabilization_per_water_eV": e_hb_per_w,
                    "fmax_tier_eV_A": tier_met if tier_met is not None else np.nan,
                    "fmax_final_eV_A": fmax_final, "nsteps_used": nsteps,
                    "converged": converged, "unphysical": unphysical,
                })

    df = pd.DataFrame(records)
    df.to_csv(RAW_CSV, index=False)
    logger.info(f"Saved raw ensemble results ({len(df)} rows) to {RAW_CSV}")

    usable = df[(df["converged"]) & (~df["unphysical"])].copy()
    n_total, n_usable = len(df), len(usable)
    logger.info(f"Usable (converged & physical) rows: {n_usable}/{n_total} "
                f"({100*n_usable/max(n_total,1):.0f}%)")

    summary = (
        usable.groupby(["qmof_id", "metal", "reaction", "state", "n_waters"])
        .agg(
            n_converged=("solvation_energy_eV", "count"),
            n_attempted=("solvation_energy_eV", lambda s: N_REPLICAS),
            mean_solvation_energy_eV=("solvation_energy_eV", "mean"),
            std_solvation_energy_eV=("solvation_energy_eV", "std"),
            mean_hb_per_water_eV=("h_bond_stabilization_per_water_eV", "mean"),
            std_hb_per_water_eV=("h_bond_stabilization_per_water_eV", "std"),
            mean_fmax_tier_eV_A=("fmax_tier_eV_A", "mean"),
        )
        .reset_index()
    )
    summary["std_solvation_energy_eV"] = summary["std_solvation_energy_eV"].fillna(0.0)
    summary["std_hb_per_water_eV"] = summary["std_hb_per_water_eV"].fillna(0.0)
    summary.to_csv(SUMMARY_CSV, index=False)
    logger.info(f"Saved aggregated ensemble summary to {SUMMARY_CSV}")

    # =========================================================================
    # FIGURE 10: Ensemble spread + convergence-tier breakdown
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax_spread, ax_conv = axes

    co_summary = summary[summary["qmof_id"] == "qmof-73ded45"]
    cu_summary = summary[summary["qmof_id"] == "qmof-b46c098"]
    state_order = ["*OH", "*O", "*OOH", "*COOH", "*CO", "*H"]
    colors_n = {1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}

    for qsum, marker, label_prefix in [(co_summary, "o", "Co-MOF-74"), (cu_summary, "s", "Cu-MOF-74")]:
        for n_w in [1, 2, 3]:
            sub = qsum[qsum["n_waters"] == n_w]
            if len(sub) == 0:
                continue
            xs = [state_order.index(s) + 0.15 * (n_w - 2) for s in sub["state"]]
            ax_spread.errorbar(
                xs, sub["mean_solvation_energy_eV"], yerr=sub["std_solvation_energy_eV"],
                fmt=marker, color=colors_n[n_w], capsize=3, markersize=7,
                label=f"{label_prefix}, n={n_w} ({sub['n_converged'].sum()}/{N_REPLICAS*len(sub)} usable)"
            )

    ax_spread.set_xticks(range(len(state_order)))
    ax_spread.set_xticklabels(state_order)
    ax_spread.axhline(0, color="k", linewidth=0.7, linestyle=":")
    ax_spread.set_ylabel(r"$\Delta E_{\mathrm{solv}}$ (eV), mean $\pm$ std over replicas")
    ax_spread.set_title(f"(a) Ensemble spread ({N_REPLICAS} random placements/level)")
    ax_spread.legend(loc="best", frameon=False, fontsize=7.5)
    ax_spread.grid(True, linestyle=":", alpha=0.5, linewidth=0.5)

    tier_counts = df.groupby("fmax_tier_eV_A", dropna=False).size()
    labels_tier = [f"{t:.2f}" if pd.notna(t) else "not conv." for t in tier_counts.index]
    ax_conv.bar(labels_tier, tier_counts.values, color="#4C72B0", edgecolor="black")
    ax_conv.set_xlabel(r"fmax tier met (eV/\AA)")
    ax_conv.set_ylabel("Number of replicas")
    ax_conv.set_title(f"(b) Convergence-tier distribution ({n_total} replicas total)")
    for i, v in enumerate(tier_counts.values):
        ax_conv.annotate(str(v), xy=(i, v), ha="center", va="bottom", fontsize=9)
    ax_conv.grid(True, linestyle=":", alpha=0.5, axis="y", linewidth=0.5)

    plt.tight_layout()
    png_path = FIG_DIR / "fig10_microsolvation_ensemble.png"
    pdf_path = FIG_DIR / "fig10_microsolvation_ensemble.pdf"
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    logger.info(f"Saved Figure 10 to {png_path} and {pdf_path}")

    # =========================================================================
    # SI TABLE 4
    # =========================================================================
    with open(TEX_OUTPUT, "w", encoding="utf-8") as f:
        f.write("% Table S4: Micro-Solvation Configurational-Sampling Ensemble\n")
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Ensemble micro-solvation energetics ($" + str(N_REPLICAS) +
                "$ independent, randomly-rotated water placements per hydration level) for the "
                "two champion MOFs. Convergence used progressively loosened fmax tiers "
                "(" + ", ".join(f"{t:.2f}" for t in FMAX_TIERS) + " eV/\\AA); $n_{\\mathrm{converged}}$ "
                "counts replicas that met some tier AND passed the $|\\Delta E_{\\mathrm{solv}}|<6$ eV "
                "physicality filter. Complements Table S2 (single deterministic geometry).}\n")
        f.write("\\label{tab:microsolvation_ensemble}\n")
        f.write("\\begin{tabular}{llccccc}\n\\hline\\hline\n")
        f.write("MOF ID & Intermediate & $n_{\\mathrm{H_2O}}$ & $n_{\\mathrm{converged}}/n_{\\mathrm{attempted}}$ & "
                "$\\langle\\Delta E_{\\mathrm{solv}}\\rangle$ (eV) & $\\sigma(\\Delta E_{\\mathrm{solv}})$ (eV) & "
                "$\\langle$tier met$\\rangle$ (eV/\\AA) \\\\\n\\hline\n")
        for _, r in summary.sort_values(["qmof_id", "state", "n_waters"]).iterrows():
            f.write(f"{r['qmof_id']} & {r['state']} & {r['n_waters']} & "
                    f"{r['n_converged']}/{N_REPLICAS} & {r['mean_solvation_energy_eV']:.3f} & "
                    f"{r['std_solvation_energy_eV']:.3f} & {r['mean_fmax_tier_eV_A']:.3f} \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n\\end{table*}\n")
    logger.info(f"Saved LaTeX SI Table S4 to {TEX_OUTPUT}")

    if RAW_CSV.exists() and SUMMARY_CSV.exists() and png_path.exists() and TEX_OUTPUT.exists():
        logger.info("\n[VALIDATION PASSED] script_13_microsolvation_ensemble.py completed successfully.")
    else:
        logger.error("\n[VALIDATION FAILED] Missing outputs in script_13.")
        sys.exit(1)


if __name__ == "__main__":
    main()
