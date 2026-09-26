# High-Throughput Mechanistic Screening of HER, OER, and CO2RR in Metal-Organic Frameworks via Equivariant Machine Learning Interatomic Potentials (MACE & CHGNet)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ASE](https://img.shields.io/badge/Powered%20by-ASE-green.svg)](https://wiki.fysik.dtu.dk/ase/)
[![MACE](https://img.shields.io/badge/MLIP-MACE-orange.svg)](https://github.com/ACEsuit/mace)
[![CHGNet](https://img.shields.io/badge/MLIP-CHGNet-red.svg)](https://github.com/CederGroupHub/chgnet)
[![GPAW](https://img.shields.io/badge/DFT-GPAW-teal.svg)](https://wiki.fysik.dtu.dk/gpaw/)

---

## 1. Scientific Overview & Objectives

Metal–Organic Frameworks (MOFs) possess exceptional structural and chemical versatility, featuring high specific surface areas, crystalline pore channels, and modular inorganic secondary building units (SBUs) interconnected by organic linkers. In electrochemical energy conversion, MOFs serve as single-site solid catalysts or tunable catalyst supports for:
1. **Hydrogen Evolution Reaction (HER):** $2\text{H}^+ + 2e^- \rightarrow \text{H}_2$
2. **Oxygen Evolution Reaction (OER):** $2\text{H}_2\text{O} \rightarrow \text{O}_2 + 4\text{H}^+ + 4e^-$
3. **$\text{CO}_2$ Reduction Reaction ($\text{CO}_2\text{RR}$):** Multi-electron reduction to $C_1$ fuels ($\text{CO}$, $\text{HCOOH}$, etc.) in direct competition with parasitic HER.

The primary objective of this project is **not merely raw high-throughput data collection**, but the establishment of a **physicochemically consistent, auditable, and reproducible computational protocol** to elucidate the fundamental reaction mechanisms, scaling relations, and selectivity principles of HER, OER, and $\text{CO}_2\text{RR}$ across the Materials Project MOF Explorer / Quantum MOF (QMOF) database (>20,000 periodic DFT structures).

---

## 2. Theoretical Framework

### 2.1 The Computational Hydrogen Electrode (CHE)
Free energy profiles are calculated within the Nørskov Computational Hydrogen Electrode (CHE) framework, setting the chemical potential of a proton-electron pair equal to half that of gaseous hydrogen:
$$\mu(\text{H}^+) + \mu(e^-) = \frac{1}{2}\mu(\text{H}_2(\text{g})) - e U + k_B T \ln(10) \cdot \text{pH}$$

The Gibbs free energy change ($\Delta G$) for each elementary electrochemical step is given by:
$$\Delta G = \Delta E_{\text{MLIP}} + \Delta \text{ZPE} - T \Delta S_{\text{vib}} + \Delta G_U + \Delta G_{\text{pH}}$$
where:
* $\Delta E_{\text{MLIP}}$ is the electronic adsorption energy computed using **MACE-MP-0** (post CHGNet pre-screening).
* $\Delta \text{ZPE}$ and $\Delta S_{\text{vib}}$ are the zero-point energy and vibrational entropy evaluated via **Partial Hessian Vibrational Analysis (PHVA)** restricted to the active site and adsorbates.

### 2.2 Modeled Intermediates & Reaction Coordinates

| Reaction | Intermediate States | Key Descriptors | Limiting Overpotential |
| :--- | :--- | :--- | :--- |
| **HER** | $* \rightarrow *\text{H} \rightarrow \text{H}_2 + *$ | $\Delta G_{*\text{H}}$ | $\eta^{\text{HER}} = \lvert \Delta G_{*\text{H}} \rvert / e$ |
| **OER** | $* \rightarrow *\text{OH} \rightarrow *\text{O} \rightarrow *\text{OOH} \rightarrow \text{O}_2 + *$ | $\Delta G_{*\text{O}} - \Delta G_{*\text{OH}}$ | $\eta^{\text{OER}} = \max[\Delta G_1, \Delta G_2, \Delta G_3, \Delta G_4]/e - 1.23\,\text{V}$ |
| **$\text{CO}_2\text{RR}$ ($\text{CO}$ path)** | $\text{CO}_2 + * \rightarrow *\text{COOH} \rightarrow *\text{CO} \rightarrow \text{CO} + *$ | $\Delta G_{*\text{COOH}}, \Delta G_{*\text{CO}}$ | $\eta^{\text{CO2RR}} = \max[\Delta G_{*\text{COOH}}, \Delta G_{*\text{CO}} - \Delta G_{*\text{COOH}}]/e - U_0$ |
| **$\text{CO}_2\text{RR}$ ($\text{HCOOH}$ path)** | $\text{CO}_2 + * \rightarrow *\text{OCHO} \rightarrow \text{HCOOH} + *$ | $\Delta G_{*\text{OCHO}}$ | $\eta^{\text{HCOOH}}$ |
| **Selectivity Metric** | $\text{CO}_2\text{RR}$ vs. parasitic HER | $\Delta G_{\text{sel}} = \Delta G_{*\text{COOH}} - \Delta G_{*\text{H}}$ | $\Delta G_{\text{sel}} < 0 \implies \text{CO}_2\text{RR}$ favored |

---

## 3. Computational Workflow & Script Pipeline

```mermaid
flowchart TD
    A["Materials Project MOF Explorer (>20,000 MOFs)"] --> B["Script 01: QMOF Filtering & SBU Metallicity Filter"]
    B --> C["Script 02: Open Metal Site (OMS/CUS) Detection & Adsorbate Placement"]
    C --> D["Script 03: Tier-1 CHGNet Screening & Framework Integrity Triage"]
    D --> E["Script 04: Tier-2 MACE-MP-0 High-Precision Relaxation (fmax < 0.02 eV/Å)"]
    E --> F["Script 05: Partial Hessian Vibrational Analysis (PHVA) & ZPE/Entropy"]
    F --> G["Script 06: CHE Free Energy Profiles & Overpotentials (HER, OER, CO2RR)"]
    G --> H["Script 07: Scaling Relations, Volcano Activity Plots & Selectivity Maps"]
    G --> I["Script 08: Local DFT Benchmark Validation via GPAW (PBE/LCAO)"]
    G --> J["Script 09: Free Energy Coordinate Profiles with Atomistic Insets"]
```

### Execution Sequence:
```bash
# Run numbered scripts sequentially
python scripts/script_01_qmof_filter.py
python scripts/script_02_site_intermediate_builder.py
python scripts/script_03_chgnet_screening.py
python scripts/script_04_mace_refinement.py
python scripts/script_05_phva_thermo.py
python scripts/script_06_che_energetics.py
python scripts/script_07_volcano_selectivity_plots.py
python scripts/script_08_gpaw_dft_validation.py
python scripts/script_09_free_energy_diagrams.py
```

---

## 4. Visual Previews & Mechanistic Analysis of Results

Below are the publication-grade figures generated by the automated protocol, establishing the scientific consistency of the calculations.

### Figure 1: OER Scaling Relation and Activity Volcano Curve
![Figure 1: OER Scaling and Volcano](figures/fig1_oer_scaling_and_volcano.png)

* **Physical Insight (Panel a):** Demonstrates a strong linear scaling relation between $\Delta G_{*\mathrm{OOH}}$ and $\Delta G_{*\mathrm{OH}}$ across diverse MOF transition metal nodes ($\Delta G_{*\mathrm{OOH}} = 0.95 \Delta G_{*\mathrm{OH}} + 3.14\,\mathrm{eV}$, $R^2 = 0.95$), in close alignment with universal metal-oxide electrocatalysis ($\Delta G_{*\mathrm{OOH}} \approx \Delta G_{*\mathrm{OH}} + 3.20\,\mathrm{eV}$).
* **Physical Insight (Panel b):** Volcano plot of overpotential $\eta^{\mathrm{OER}}$ versus the descriptor $(\Delta G_{*\mathrm{O}} - \Delta G_{*\mathrm{OH}})$. Open metal sites in Co-MOF-74 (`qmof-73ded45`, $\eta = 0.81\,\mathrm{V}$) and Mo-MOF (`qmof-04b4379`, $\eta = 0.75\,\mathrm{V}$) lie closest to the theoretical summit.

---

### Figure 2: HER Sabatier Volcano Activity Curve
![Figure 2: HER Volcano](figures/fig2_her_volcano.png)

* **Physical Insight:** Theoretical Sabatier volcano activity plot ($-\eta^{\mathrm{HER}}$ vs. $\Delta G_{*\mathrm{H}}$). Maximum exchange current density corresponds to termoneutral hydrogen binding ($\Delta G_{*\mathrm{H}} = 0\,\mathrm{eV}$).
* **Top Candidates:** Cu-MOF-74 (`qmof-b46c098`, $\eta = 0.12\,\mathrm{V}$), Mo-MOF (`qmof-04b4379`, $\eta = 0.11\,\mathrm{V}$), and Ni-MOF (`qmof-da6b9c1`, $\eta = 0.11\,\mathrm{V}$) reside right at the apex of the volcano curve.

---

### Figure 3: $\text{CO}_2\text{RR}$ vs. Parasitic HER Selectivity Map
![Figure 3: CO2RR vs HER Selectivity](figures/fig3_co2rr_her_selectivity.png)

* **Physical Insight (Panel a):** Two-dimensional selectivity map comparing $\Delta G_{*\mathrm{COOH}}$ against $\Delta G_{*\mathrm{H}}$. The diagonal line ($\Delta G_{*\mathrm{COOH}} = \Delta G_{*\mathrm{H}}$) demarcates the HER-dominant regime (red) from the $\mathrm{CO}_2\mathrm{RR}$-selective regime (green). Most pristine transition metal open sites favor proton reduction, emphasizing the necessity of microenvironment linker tailoring.
* **Physical Insight (Panel b):** Gibbs free energy reaction coordinate diagram for the $C_1 \rightarrow \mathrm{CO}$ reduction pathway ($\mathrm{CO}_2 \rightarrow *\mathrm{COOH} \rightarrow *\mathrm{CO} \rightarrow \mathrm{CO}$) across representative metal centers at standard state ($U = 0\,\mathrm{V}$).

---

### Figure 4: OER Reaction Coordinate Free Energy Diagram with Atomistic Insets
![Figure 4: OER Free Energy Diagram](figures/fig4_oer_free_energy_diagram.png)

* **Physical Insight:** Four-electron associative OER path ($* \rightarrow *\mathrm{OH} \rightarrow *\mathrm{O} \rightarrow *\mathrm{OOH} \rightarrow \mathrm{O}_2$) on Co-MOF-74 (`qmof-73ded45`) evaluated at:
  1. $U = 0.00\,\mathrm{V}$ (red profile, all steps endergonic)
  2. $U = 1.23\,\mathrm{V}$ (blue profile, standard equilibrium potential)
  3. $U = U_L = 2.04\,\mathrm{V}$ (green profile, onset potential where all elementary steps become downhill, $\eta = 0.81\,\mathrm{V}$).
* **Atomistic Insets:** High-contrast renderings of the pristine Co open metal node, $*OH$ adduct, $*O$ oxo intermediate, and $*OOH$ hydroperoxyl intermediate.

---

### Figure 5: HER Reaction Coordinate Free Energy Diagram with Atomistic Insets
![Figure 5: HER Free Energy Diagram](figures/fig5_her_free_energy_diagram.png)

* **Physical Insight:** Two-electron HER pathway on Cu-MOF-74 (`qmof-b46c098`) at $U = 0.00\,\mathrm{V}$ and at the onset overpotential $U = -0.12\,\mathrm{V}$, demonstrating near-ideal Sabatier termoneutrality with an overpotential of only $0.12\,\mathrm{V}$.
* **Atomistic Insets:** Localized active site cluster showing the uncoordinated pristine Cu site and the chemically relaxed Cu-H adduct ($d_{\mathrm{Cu-H}} \approx 1.55\,\text{\AA}$).

---

### Figure 6: $\text{CO}_2\text{RR} \rightarrow \text{CO}$ Reaction Coordinate Free Energy Diagram with Atomistic Insets
![Figure 6: CO2RR Free Energy Diagram](figures/fig6_co2rr_free_energy_diagram.png)

* **Physical Insight:** Two-electron $\mathrm{CO}_2$ reduction to $\mathrm{CO}$ on Mn-MOF (`qmof-07cc468`) at $U = 0.00\,\mathrm{V}$ and onset potential $U = U_L$, identifying the potential-determining step (PDS) between $*COOH$ formation and $*CO$ desorption.
* **Atomistic Insets:** Atomistic configurations of the open Mn center, the carbon-bound carboxyl intermediate ($*COOH$), and the linear carbonyl adduct ($*CO$).

---

### Figure 8: Micro-Solvation & Hydrogen-Bonding Network Scaling Breakdown Analysis
![Figure 8: Micro-Solvation Scaling Breakdown](figures/fig8_microsolvation_scaling_break.png)

* **Panel (a) — Scaling Relation Breaking:** Evolution of $\Delta G_{*\mathrm{OOH}}$ vs $\Delta G_{*\mathrm{OH}}$ in Co-MOF-74 (`qmof-73ded45`) as hydration degree increases from dry ($n=0$) to solvated ($n=3\,\mathrm{H_2O}$). Asymmetry in hydrogen bonding selectively stabilizes $*OOH$ by $-2.53\,\mathrm{eV}$ (via a cooperative 6-membered cyclic H-bond network) versus only $-0.65\,\mathrm{eV}$ for $*OH$, collapsing the universal scaling gap from $3.20\,\mathrm{eV}$ down to $2.74\,\mathrm{eV}$ toward the ideal catalyst line ($2.46\,\mathrm{eV}$).
* **Panel (b) — OER Overpotential Suppression:** Four-electron free energy profiles at standard potential $U = 1.23\,\mathrm{V}$. Micro-solvation lowers the activation barrier of the potential-determining step ($*OH \rightarrow *O$), suppressing the OER overpotential from $\eta = 0.81\,\mathrm{V}$ to $\eta = 0.42\,\mathrm{V}$.
* **Panel (c) — $\mathrm{CO}_2\mathrm{RR}$ vs Parasitic HER Selectivity:** Progression of $\Delta G_{*\mathrm{COOH}}$ vs $\Delta G_{*\mathrm{H}}$ with pore water molecules. Polar carboxyl groups ($-COOH$) experience strong multi-point hydration ($\Delta E_{\mathrm{solv}} = -0.63\,\mathrm{eV}$), whereas non-polar hydride ($*H$) has minimal solvation, expanding the selectivity gap in favor of $\mathrm{CO}_2$ reduction.
* **Panel (d) — H-Bond Stabilization Energy:** Differential stabilization energy ($\Delta E_{\mathrm{HB}}$ per $\mathrm{H_2O}$) across all key intermediates ($*OH$, $*O$, $*OOH$, $*COOH$, $*CO$, $*H$).

---

### Figure 9: High-Throughput Electrocatalytic Scale-Up Across 9 Transition Metal Families
![Figure 9: High-Throughput Scale-Up](figures/fig9_high_throughput_scaling_distributions.png)

* **Panel (a) — Cohort-Wide HER Sabatier Volcano:** Evaluation of hydrogen evolution overpotentials ($\eta^{\mathrm{HER}}$ vs $\Delta G_{*\mathrm{H}}$) across 32 synthesized MOFs spanning Co, Cu, Fe, Mn, Mo, Ni, Ru, Zn, and Zr. Cu- and Mo-based frameworks consistently optimize the thermoneutral region near $\Delta G_{*\mathrm{H}} \approx 0\,\mathrm{eV}$.
* **Panel (b) — Cross-Metal OER Scaling Relations:** $\Delta G_{*\mathrm{OOH}}$ vs $\Delta G_{*\mathrm{OH}}$ linear correlation across varied framework topologies and transition metal active nodes ($\Delta G_{*\mathrm{OOH}} = 0.94\Delta G_{*\mathrm{OH}} + 3.15\,\mathrm{eV}$), demonstrating the universal persistence of scaling limits in the absence of secondary coordination sphere/solvation engineering.
* **Panel (c) — 2D $\mathrm{CO}_2\mathrm{RR}$ vs HER Selectivity Landscape:** Global mapping of $\Delta G_{*\mathrm{COOH}}$ vs $\Delta G_{*\mathrm{H}}$ across the 32 MOF library, demarcating materials favoring selective carbon dioxide activation from those dominated by parasitic hydride formation.
* **Panel (d) — Structural-Electronic Property Correlation:** Multivariate mapping of OER overpotential as a joint function of Pore Limiting Diameter (PLD) and electronic DFT bandgap ($E_g$). Nanoporous channels between $6.0$ and $11.0\,\text{\AA}$ and moderate bandgaps ($1.0 - 2.5\,\text{eV}$) achieve optimal catalytic activity.

---

## 5. Software & Citations

1. **CHGNet:** Deng et al., *Nature Machine Intelligence* **5** (2023) 1031–1041.
2. **MACE:** Batatia et al., *Advances in Neural Information Processing Systems* (NeurIPS) **35** (2022) 11423–11436.
3. **QMOF Database:** Rosen et al., *Chemistry of Materials* **33** (2021) 7226–7237.
4. **Materials Project:** Jain et al., *APL Materials* **1** (2013) 011002.
5. **ASE:** Larsen et al., *J. Phys.: Condens. Matter* **29** (2017) 273002.
6. **GPAW:** Enkovaara et al., *J. Phys.: Condens. Matter* **22** (2010) 253202.
7. **Pymatgen:** Ong et al., *Computational Materials Science* **68** (2013) 314–319.
8. **Nørskov CHE Model:** Nørskov et al., *J. Phys. Chem. B* **108** (2004) 17886–17892.

---
*Maintained by Prof. Luiz Antonio Ribeiro Junior — LCCMat / UnB & NTNU (2026).*
