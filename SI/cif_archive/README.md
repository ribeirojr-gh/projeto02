# Supporting Information: Structure Archive

Every structure analysed in this study, organised by study phase. Copied
from the working `structures/` tree used by the analysis pipeline
(`scripts/script_01`-`script_13`); nothing here was recomputed, this is a
reorganisation and tabulation of existing results for Supporting
Information purposes.

| # | Phase | N structures | Folder | Table |
|---|-------|-------------:|--------|-------|
| 1 | Benchmark cohort (18 MOFs), pristine frameworks | 18 | `01_benchmark_cohort_pristine/` | Table S5 |
| 2 | Benchmark cohort, Tier-1 CHGNet pre-relaxed (pristine + 7 intermediates x 18) | 144 | `02_benchmark_cohort_tier1_chgnet_prerelaxed/` | -- (provenance only) |
| 3 | Benchmark cohort, Tier-2 MACE-MP-0 relaxed -- **FINAL production structures** | 144 | `03_benchmark_cohort_tier2_mace_relaxed_FINAL/` | Table S6 |
| 4 | High-throughput scale-up cohort (32 MOFs), pristine frameworks | 32 | `04_scaleup_cohort_pristine_32MOF/` | Table S9 |
| 5 | Micro-solvation, single deterministic geometry (2 champions, n=1,2,3 H2O) | 44 | `05_microsolvation_single_geometry/` | Table S2 |
| 6 | Micro-solvation, configurational-sampling ensemble (5 replicas/level) | 135 | `06_microsolvation_ensemble/` | Table S4 |

**Total: 517 CIF files.**

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
