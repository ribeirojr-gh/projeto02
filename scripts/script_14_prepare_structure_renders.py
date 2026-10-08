#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 14: Prepare Representative-Structure Render Inputs
Project: mofs-mace-her-oer-co2rr (Repo: projeto02-26092026)

Builds the local structural inputs (PDB files) for PyMOL snapshot rendering
of the three reaction champions (framework overview + active-site close-up
with the bound intermediate highlighted) and one micro-solvation example
(active site + explicit pore water highlighted separately from the
chemisorbed intermediate).

Why not just re-use the saved CIF files directly: ASE's CIF reader does not
guarantee preserving the original atom order on round-trip (observed:
atoms get re-grouped, not kept in [framework..., adsorbate..., water...]
append order as originally written by the simulation scripts). Figuring out
which atoms are "the adsorbate" or "the water" by position-in-list after a
CIF round-trip is therefore unreliable. Instead, this script identifies new
atoms geometrically: per chemical element, it Hungarian-matches each atom
in a smaller reference structure (e.g. the pristine framework) to its
closest-fitting counterpart in a larger structure (e.g. the same framework
plus an adsorbate); the few atoms of the larger structure left unmatched
are, by construction, the atoms that do not exist in the reference -- i.e.
the adsorbate (or, one level up, the explicit water).

Dependencies: ase, numpy, scipy
"""

import sys
import logging
from pathlib import Path
import numpy as np
import ase.io
from ase import Atoms
from ase.geometry import get_distances
from scipy.optimize import linear_sum_assignment

BASE_DIR = Path(__file__).resolve().parent.parent
MACE_DIR = BASE_DIR / "structures" / "mace_relaxed"
PRISTINE_DIR = BASE_DIR / "structures" / "pristine_mofs"
SOLV_DIR = BASE_DIR / "structures" / "solvated_intermediates"
OUT_DIR = BASE_DIR / "structures" / "render_inputs"
LOG_DIR = BASE_DIR / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "results.log", mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("script_14_prepare_structure_renders")


def match_new_atoms(ref_atoms: Atoms, query_atoms: Atoms):
    """Per-element Hungarian matching of ref_atoms into query_atoms.
    Returns (matched_indices_in_query, new_indices_in_query)."""
    matched = []
    cell = query_atoms.cell
    pbc = query_atoms.pbc
    ref_syms = ref_atoms.get_chemical_symbols()
    qry_syms = query_atoms.get_chemical_symbols()
    for element in sorted(set(ref_syms)):
        ref_idx = [i for i, s in enumerate(ref_syms) if s == element]
        qry_idx = [i for i, s in enumerate(qry_syms) if s == element]
        if len(qry_idx) < len(ref_idx):
            raise ValueError(f"query has fewer {element} atoms ({len(qry_idx)}) than ref ({len(ref_idx)})")
        ref_pos = ref_atoms.positions[ref_idx]
        qry_pos = query_atoms.positions[qry_idx]
        _, D = get_distances(ref_pos, qry_pos, cell=cell, pbc=pbc)
        row, col = linear_sum_assignment(D)
        for r, c in zip(row, col):
            matched.append(qry_idx[c])
    matched = sorted(set(matched))
    new = sorted(set(range(len(query_atoms))) - set(matched))
    return matched, new


def unwrap_structure(atoms: Atoms, seed_idx: int = 0, bond_cutoff: float = 2.6, search_range: int = 2) -> dict:
    """Returns {orig_idx: physically-unwrapped position} for the WHOLE
    structure, built by flood-filling outward from seed_idx: each newly
    reached atom is placed at whichever periodic image is closest to the
    neighbour that discovered it (not to an external fixed point), so every
    bonded pair ends up at its true bond distance, transitively, across the
    whole connected network -- this is what a naive "closest image to one
    fixed centre" approach gets wrong whenever the structure is wrapped such
    that two bonded atoms individually sit far from that centre's image.
    bond_cutoff=2.6 A covers covalent C-C/C-O/O-H bonds and typical metal-O/
    metal-C coordination bonds in this system without being so large it
    bridges genuinely non-bonded close contacts.
    A structure can include atoms NOT reachable by bonds from the seed (e.g.
    a dissociated fragment, observed for the *COOH state on qmof-dc7e5a3):
    any such leftover component is flood-filled internally on its own (so
    its own internal geometry is still self-consistent), then rigidly
    translated by a whole number of lattice vectors to the shift that
    brings it closest to the already-placed structure, purely so it renders
    near the rest of the cluster rather than kilometres away in fractional-
    coordinate space."""
    from collections import deque
    n = len(atoms)
    cell = atoms.cell[:]
    shifts = np.array([da * cell[0] + db * cell[1] + dc * cell[2]
                        for da in range(-search_range, search_range + 1)
                        for db in range(-search_range, search_range + 1)
                        for dc in range(-search_range, search_range + 1)])
    placed = {}
    remaining = set(range(n))

    def bfs_from(seed):
        placed[seed] = atoms.positions[seed].copy()
        remaining.discard(seed)
        queue = deque([seed])
        while queue:
            p = queue.popleft()
            p_pos = placed[p]
            for q in list(remaining):
                candidates = atoms.positions[q] + shifts
                d = np.linalg.norm(candidates - p_pos, axis=1)
                j = int(np.argmin(d))
                if d[j] <= bond_cutoff:
                    placed[q] = candidates[j]
                    remaining.discard(q)
                    queue.append(q)

    bfs_from(seed_idx)
    while remaining:
        before = set(placed.keys())
        bfs_from(next(iter(remaining)))
        new_idx = list(set(placed.keys()) - before)
        main_pos = np.array([placed[i] for i in before])
        comp_pos = np.array([placed[i] for i in new_idx])
        best_shift, best_d = np.zeros(3), np.inf
        for da in range(-search_range, search_range + 1):
            for db in range(-search_range, search_range + 1):
                for dc in range(-search_range, search_range + 1):
                    shift = da * cell[0] + db * cell[1] + dc * cell[2]
                    dmin = np.linalg.norm((comp_pos + shift)[:, None, :] - main_pos[None, :, :], axis=2).min()
                    if dmin < best_d:
                        best_d, best_shift = dmin, shift
        for i in new_idx:
            placed[i] = placed[i] + best_shift

    return placed


def select_cluster(unwrapped: dict, atoms: Atoms, center_pos: np.ndarray, cutoff: float):
    """Simple Euclidean cutoff-sphere selection from an already fully
    unwrapped (self-consistent, cell-free) structure."""
    kept_positions, kept_symbols, orig_idx_map = [], [], []
    for i, pos in unwrapped.items():
        if np.linalg.norm(pos - center_pos) <= cutoff:
            kept_positions.append(pos)
            kept_symbols.append(atoms[i].symbol)
            orig_idx_map.append(i)
    cluster = Atoms(symbols=kept_symbols, positions=kept_positions)
    return cluster, orig_idx_map


def write_labelled_pdb(cluster: Atoms, role_by_pos: list, out_path: Path):
    """Writes `cluster` to PDB with a per-atom role recorded in the B-factor column
    (0=framework, 1=adsorbate, 2=water) so PyMOL can select by role without needing
    separate files; also writes separate per-role PDBs for simplicity of use."""
    bfac = np.array(role_by_pos, dtype=float)
    cluster_copy = cluster.copy()
    cluster_copy.set_array("role", bfac)
    ase.io.write(str(out_path.parent / f"{out_path.stem}_full.pdb"), cluster_copy, format="proteindatabank")
    # ASE's PDB writer does not carry arbitrary arrays into the B-factor column
    # reliably across versions, so also write role-split files explicitly:
    for role_id, role_name in [(0, "framework"), (1, "adsorbate"), (2, "water")]:
        idx = [i for i, r in enumerate(role_by_pos) if r == role_id]
        if idx:
            ase.io.write(str(out_path.parent / f"{out_path.stem}_{role_name}.pdb"), cluster[idx], format="proteindatabank")


def main():
    logger.info("=== STEP 14: Prepare Representative-Structure Render Inputs ===")

    champions = [
        # (label, metal, pristine_cif, decorated_cif, n_ads, framework_repeat)
        # All three share the MOF-74 topology: short a-axis (~6.3-6.8 A) runs
        # along the 1D pore channel, b/c (~15 A) span across it. Repeating
        # only along a gives a clean channel segment instead of a cluttered
        # 3D block.
        ("oer_champion_CoMOF74", "Co", "qmof-73ded45.cif",
         MACE_DIR / "qmof-73ded45_oer_OH.cif", (3, 1, 1)),
        ("her_champion_CuMOF74", "Cu", "qmof-b46c098.cif",
         MACE_DIR / "qmof-b46c098_her_H.cif", (3, 1, 1)),
        ("co2rr_champion_qmofdc7e5a3", "Co", "qmof-dc7e5a3.cif",
         MACE_DIR / "qmof-dc7e5a3_co2rr_COOH.cif", (3, 1, 1)),
    ]

    for label, metal, pristine_name, decorated_path, framework_repeat in champions:
        pristine_path = PRISTINE_DIR / pristine_name
        pristine = ase.io.read(str(pristine_path))
        decorated = ase.io.read(str(decorated_path))

        # 1) Framework overview supercell (pristine, no adsorbate)
        sc = pristine.repeat(framework_repeat)
        sc.set_pbc(False)
        sc.set_cell(None)
        ase.io.write(str(OUT_DIR / f"{label}_framework_overview.pdb"), sc, format="proteindatabank")

        # 2) Active-site close-up: split decorated structure into framework/adsorbate
        matched, new = match_new_atoms(pristine, decorated)
        logger.info(f"{label}: decorated={len(decorated)} atoms, framework-matched={len(matched)}, "
                    f"adsorbate(new)={len(new)} -> symbols {[decorated[i].symbol for i in new]}")
        # Unwrap the WHOLE structure (flood-fill from a metal seed) so every
        # bonded pair sits at its true distance, then take a simple
        # Euclidean cutoff sphere around the (now self-consistently placed)
        # adsorbate centroid.
        metal_seed = next(i for i, s in enumerate(decorated.get_chemical_symbols()) if s == metal)
        unwrapped = unwrap_structure(decorated, seed_idx=metal_seed)
        ads_centre = np.mean([unwrapped[i] for i in new], axis=0)
        cluster, orig_idx_map = select_cluster(unwrapped, decorated, ads_centre, cutoff=7.0)
        role = [1 if oi in new else 0 for oi in orig_idx_map]
        logger.info(f"  active-site cluster: {len(cluster)} atoms, adsorbate-role count={sum(role)}")
        write_labelled_pdb(cluster, role, OUT_DIR / f"{label}_active_site")

    # 3) Micro-solvation example: Co-MOF-74, *OH, n=2 H2O
    pristine = ase.io.read(str(PRISTINE_DIR / "qmof-73ded45.cif"))
    decorated_oh = ase.io.read(str(MACE_DIR / "qmof-73ded45_oer_OH.cif"))
    solvated = ase.io.read(str(SOLV_DIR / "qmof-73ded45_OH_solv2w.cif"))

    matched_fw, new_vs_pristine = match_new_atoms(pristine, solvated)   # new = OH + waters
    matched_oh, new_vs_oh = match_new_atoms(decorated_oh, solvated)     # matched = framework+OH, new = waters
    adsorbate_idx = sorted(set(new_vs_pristine) - set(new_vs_oh))
    water_idx = sorted(new_vs_oh)
    logger.info(f"solvation example: total={len(solvated)}, framework={len(matched_oh) - len(adsorbate_idx)}, "
                f"adsorbate(OH)={len(adsorbate_idx)} symbols={[solvated[i].symbol for i in adsorbate_idx]}, "
                f"water={len(water_idx)} symbols={[solvated[i].symbol for i in water_idx]}")

    co_seed = next(i for i, s in enumerate(solvated.get_chemical_symbols()) if s == "Co")
    unwrapped = unwrap_structure(solvated, seed_idx=co_seed)
    centre_idx = adsorbate_idx + water_idx
    solv_centre = np.mean([unwrapped[i] for i in centre_idx], axis=0)
    cluster, orig_idx_map = select_cluster(unwrapped, solvated, solv_centre, cutoff=7.0)
    role = []
    for oi in orig_idx_map:
        if oi in adsorbate_idx:
            role.append(1)
        elif oi in water_idx:
            role.append(2)
        else:
            role.append(0)
    logger.info(f"  solvation cluster: {len(cluster)} atoms, role counts: framework={role.count(0)}, "
                f"adsorbate={role.count(1)}, water={role.count(2)}")
    write_labelled_pdb(cluster, role, OUT_DIR / "solvation_example_CoMOF74_OH_2H2O_active_site")

    logger.info("[VALIDATION] Files written:")
    for p in sorted(OUT_DIR.glob("*.pdb")):
        logger.info(f"  {p.name}")
    logger.info("[VALIDATION PASSED] script_14_prepare_structure_renders.py completed successfully.")


if __name__ == "__main__":
    main()
