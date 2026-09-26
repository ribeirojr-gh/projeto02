#!/usr/bin/env python
# coding: utf-8

"""
Exciton radiative lifetime from Yambo BSE outputs
Based on:
Palummo et al., Nano Letters 15, 2794 (2015)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

plt.style.use("classic")

# ============================================================
# Plot style
# ============================================================
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{mathptmx}",
    "font.size": 10,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5
})

# ==================================================
# Fundamental constants
# ==================================================
c = 299792458.0                 # m/s
a0 = 5.2917721067e-11           # Bohr radius (m)
me = 9.1093837e-31              # electron mass (kg)
Ha2eV = 27.21
j_to_ev = 6.241509e18
kB = 8.6173303e-5               # eV/K
pi = np.pi

# ==================================================
# USER INPUT
# ==================================================
spin = 2                        # spin degeneracy
exciton_mass = 0.66             # M_s = m_e + m_h (in units of m_e)

# Exciton groups (1-based indices)
exciton_groups = {
    "1s"   : [1],
    "2p"   : [5,6],
    "2s"     : [7],
    "3d"  : [15, 16],
    "3p"  : [19]
}


# Temperatures
T_list = [0.0, 4.0, 300.0]      # values to print
T_plot = np.linspace(5, 350, 300)

# ==================================================
# INPUT FILES
# ==================================================
path = os.getcwd()
path_energies = os.path.join(path, "o-BSE.exc_qpt1_E_sorted")
path_rsetup   = os.path.join(path, "r_setup")

# ==================================================
# READ DATA
# ==================================================
exciton_energy, osc_strength = np.genfromtxt(
    path_energies, skip_header=55, usecols=[0, 1], unpack=True
)

with open(path_rsetup) as f:
    for line in f:
        if "Direct lattice volume" in line:
            Volume_uc = float(line.split()[4])
        if "Alat factors" in line:
            lattice_c = float(line.split()[5])

with open(path_energies) as f:
    for line in f:
        if "Maximum Residual Value" in line:
            Residuals = float(line.split()[5])

Area_uc = Volume_uc / lattice_c

# ==================================================
# Thermal radiative rate for a group of excitons
# ==================================================
T_LT_cutoff = 5.0   # LT regime cutoff in K (used for 4 K physics)

def gamma_thermal_group(T, indices):
    """
    Effective radiative rate for a group of excitons
    following Palummo et al.
    indices: list of 1-based exciton indices
    """

    idx = np.array(indices) - 1
    E = exciton_energy[idx]
    f = osc_strength[idx]

    # dipole matrix elements
    mu2 = Residuals * f * Volume_uc / (spin * 4 * pi * Ha2eV)

    # gamma at T = 0
    gamma0 = (
        4 * pi * E * mu2
        / (me * c * a0 * j_to_ev * Area_uc)
    )

    # Strict T = 0 K
    if T == 0.0:
        return float(np.sum(gamma0))

    # Low-temperature regime (no thermal averaging)
    if T < T_LT_cutoff:
        prefactor = (
            4 * E**2
            / (3 * 2 * me * c**2 * j_to_ev * exciton_mass * kB * T)
        )
        return float(np.sum(gamma0 * prefactor))

    # Room-temperature regime (Boltzmann average)
    prefactor = (
        4 * E**2
        / (3 * 2 * me * c**2 * j_to_ev * exciton_mass * kB * T)
    )
    gamma_T = gamma0 * prefactor

    # shift energies for numerical stability
    weights = np.exp(-(E - E.min()) / (kB * T))

    return float(np.sum(gamma_T * weights) / np.sum(weights))

# ==================================================
# Compute γ(T) and τ(T) for all exciton groups
# ==================================================
gamma_groups = {}
tau_groups   = {}

for label, indices in exciton_groups.items():
    gamma_groups[label] = np.array(
        [gamma_thermal_group(T, indices) for T in T_plot]
    )
    tau_groups[label] = 1.0 / gamma_groups[label]

# ==================================================
# Print selected temperatures
# ==================================================
print("\n=== Radiative lifetimes ===\n")
for label, indices in exciton_groups.items():
    print(f"{label}")
    for T in T_list:
        gT = gamma_thermal_group(T, indices)
        tT = 1.0 / gT
        print(f"  T = {T:6.1f} K :  gamma = {gT:9.3e} s^-1   tau = {tT*1e12:8.3f} ps")
    print()

# ==================================================
# PLOTS
# ==================================================
colors = {
    "1s"  : "tab:purple",
    "2p"  : "tab:orange",
    "2s"    : "tab:blue",
    "3d" : "tab:red",
    "3p" : "tab:green",
}

# ---- Radiative rate ----
#plt.figure(figsize=(7,5))
#for label in exciton_groups:
#    plt.plot(T_plot, gamma_groups[label] * 1e-9, lw=2, color=colors[label], label=label)

#plt.xlabel("Temperature [K]", fontsize=14)
#plt.ylabel(r"Radiative rate $\gamma$ [ns$^{-1}$]", fontsize=14)
#plt.legend(fontsize=11)
#plt.tight_layout()
#plt.show()

# ---- Radiative lifetime ----
plt.figure(figsize=(4,2.8))
for label in exciton_groups:
    plt.plot(T_plot, tau_groups[label] * 1e9, lw=1.2,color=colors[label], label=label)

plt.xlabel("Temperature [K]")
plt.ylabel(r"Exciton radiative lifetime [ns]")
plt.yscale("log")
plt.ylim(1e-3, None)
plt.legend(fontsize=8, loc="lower right",frameon=False)
plt.tick_params(which="both", direction="out")
plt.tight_layout()
plt.savefig("radiative-lifetime_T.svg", dpi=300)
plt.savefig("FIG4-R1_radiative-lifetime_T.png", dpi=500)
plt.show()
