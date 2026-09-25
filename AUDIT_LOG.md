# AUDIT LOG: Computational Protocol for HER, OER, and CO2RR on MOFs

## Protocol Metadata
* **Project Name:** `mofs-mace-her-oer-co2rr`
* **PI / Author:** Prof. Luiz Antonio Ribeiro Junior (LCCMat / UnB & NTNU)
* **Date Initialized:** 2026-09-25
* **Execution Environment:** Linux x86_64, Python 3.12, CUDA-accelerated PyTorch (NVIDIA GeForce RTX 4070 Laptop GPU)
* **Primary Frameworks:** Atomic Simulation Environment (ASE), Pymatgen, CHGNet (GNN-MP), MACE (E(3)-Equivariant Message Passing)
* **Databases:** Materials Project MOF Explorer / QMOF Database (>20,000 DFT-relaxed MOFs)
* **API Key Auth:** Materials Project REST API validated

---

## Audit Trail of Actions & Validations

### [2026-09-25 18:00] Step 0: Protocol Approval & Environment Certification
* **Action:** Confirmed simulation proposal with user.
* **Storage Audit:**
  - Google Drive: Verified write/read access to `GEMINI-APPLICATIONS/mofs-her-oer-co2rr` via GVFS.
  - GitHub: Connected to `ribeirojr-gh/mofs-mace-her-oer-co2rr`, cloned locally, initialized `develop` branch.
* **Package Diagnostics:**
  - Resolved NumPy C-extension dependency conflict by updating `numexpr` and `bottleneck` to NumPy 2-compatible wheels.
  - Verified import and CUDA capability for `CHGNet` and `MACE-MP-0`.
* **Deliverable:** `sync_to_gdrive.py`, `environment.yml`, `AUDIT_LOG.md`.
* **Status:** PASS.

### [2026-09-25 19:15] Step 1 & 2: Database Ingestion, Screening & Structure Extraction
* **Action:** Executed `scripts/script_01_qmof_filter.py`.
* **Database Queried:** Materials Project MOF Explorer / QMOF Database (`qmof_database.zip`, 20,372 entries).
* **Filters Applied:**
  - Synthesized experimentally: `True`.
  - Target transition metal nodes: Co, Cu, Fe, Mn, Mo, Ni, Ru, Zn, Zr.
  - Pore Limiting Diameter (PLD) $\ge 2.5\text{ \AA}$ (diffusion accessibility for $\text{H}_2\text{O}$ and $\text{CO}_2$).
  - Tractable atom count: $N_{\text{atoms}} \le 200$.
* **Results:**
  - Screened pool: 1,514 electrocatalytically relevant MOFs exported to `data/metadata_qmof_filtered.csv`.
  - Selected benchmark cohort: 18 representative MOFs covering 9 transition metal families (including open metal site archetypes like MOF-74 analogues) saved to `data/cohort_benchmark_mofs.csv`.
  - Extracted & validated 18 pristine CIF structures to `structures/pristine_mofs/`.
* **Deliverable:** `scripts/script_01_qmof_filter.py`, `data/metadata_qmof_filtered.csv`, `data/cohort_benchmark_mofs.csv`, 18 pristine `.cif` files.
* **Status:** PASS ([VALIDATION PASSED] logged).

### [2026-09-25 19:25] Step 4: Open Metal Site Detection & Adsorbate Placement
* **Action:** Executed `scripts/script_02_site_intermediate_builder.py`.
* **Methodology:**
  - Evaluated the first coordination sphere around each transition metal node using a radial cutoff $R = 2.5\text{ \AA}$.
  - Computed the outward-pointing open coordination vector ($\hat{u}_{\text{open}}$) from inverted normalized ligand vectors.
  - Placed 7 reaction intermediates along $\hat{u}_{\text{open}}$:
    * **HER:** $*H$ ($d_{\text{M-H}} = 1.55\text{ \AA}$)
    * **OER:** $*OH$ ($1.85\text{ \AA}$), $*O$ ($1.68\text{ \AA}$), $*OOH$ ($1.85\text{ \AA}$ with $\angle\text{MOO} = 110^\circ$)
    * **$\text{CO}_2\text{RR}$:** $*COOH$ ($1.95\text{ \AA}$, C-bound), $*CO$ ($1.85\text{ \AA}$, C-bound), $*OCHO$ ($1.95\text{ \AA}$, O-bound)
  - Handled space-group symmetry breaking via ASE P1-preserving serialization.
* **Validation:** Verified all 126 structures for physical binding distance ($1.2\text{ \AA} \le d \le 2.6\text{ \AA}$) with zero steric clashes.
* **Deliverable:** `scripts/script_02_site_intermediate_builder.py`, `data/active_sites_summary.csv`, 126 `.cif` structures across `structures/intermediates_{her,oer,co2rr}/`.
* **Status:** PASS ([VALIDATION PASSED] logged).


