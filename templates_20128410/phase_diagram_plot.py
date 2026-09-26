import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

# ============================================================
# Colors
# ============================================================
c_blue   = "#b5d8ef"
c_yellow = "#eeedbb"
c_grey   = "#e1e5e8"
c_bcs    = "lightsalmon"
c_purple = "tab:purple"
c_red        = "#B2182B"   # deep red
c_orange     = "#EF8A00"   # strong orange
c_sand       = "#E6BE6A"   # warm sand
c_pale_yellow= "#F2E8B6"   # pale yellow
c_light_green= "#A6C96A"   # light green
c_mid_green  = "#5AA14F"   # medium green
c_dark_green = "#0B6B2E"   # dark green
# ============================================================
# Physical constants
# ============================================================
kB   = 1.380649e-23
hbar = 1.054571817e-34
m0   = 9.10938356e-31

# ============================================================
# Input parameters
# ============================================================
m_exc = 0.66
EBE   = 0.724
Reff  = 9.87  # Å

m_ex = m_exc * m0

# Dissociation temperature
kB_eV = 8.617333262e-5
Tc = 0.1 * EBE / kB_eV

#Room-temperature density
T_RT = 300  # in K
n_RT = (kB * T_RT * m_ex) / (2 * np.pi *hbar**2)

# Critical densities
Reff_cm = Reff * 1e-8
n_D = np.exp(-np.e) / Reff_cm**2
n_M = 1.0 / (np.pi * Reff_cm**2)

n_D_plot = n_D / 1e12
n_M_plot = n_M / 1e12
n_RT_plot = n_RT / (1e12 *1e4)

# Critical temperature at n_D
n_D_SI = n_D * 1e4
Td_BKT = (np.pi * hbar**2 / (2 * kB * m_ex)) * n_D_SI
Td_BEC = (2 * np.pi * hbar**2 / (kB * m_ex)) * n_D_SI
# ============================================================
# Print parameters
# ============================================================
print("===== Exciton BEC results =====")
print(f"T_c = {Tc:.2f} K")
print(f"T_d for BEC = {Td_BEC:.2f} K")
print(f"T_d for BKT = {Td_BKT:.2f} K")
print(f"n_RT (n at T = 300 K) = {n_RT_plot:.2f} × 10^12 cm^-2")
print(f"n_D = {n_D_plot:.2f} × 10^12 cm^-2")
print(f"n_M = {n_M_plot:.2f} × 10^12 cm^-2")
print("===============================================")

# ============================================================
# Density grids
# ============================================================
n_left = np.linspace(0, n_D_plot, 400)
n_bcs  = np.linspace(n_D_plot, n_M_plot, 400)

# Convert to SI
n_left_SI = n_left * 1e12 * 1e4

# BKT (left side only)
T_BKT_left = (np.pi * hbar**2 / (2 * kB * m_ex)) * n_left_SI
T_BEC_left = (2 * np.pi * hbar**2 / (kB * m_ex)) * n_left_SI

# ============================================================
# Decreasing BCS boundary (manual linear interpolation)
# ============================================================
T_BCS_boundary = Td_BKT * (n_M_plot - n_bcs) / (n_M_plot - n_D_plot)

# ============================================================
# Plot limits
# ============================================================
n_max = 1.25 * n_M_plot
Tmax  = 900

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(4, 4))
#c_red        = "#B2182B"   # deep red
#c_orange     = "#EF8A00"   # strong orange
#c_sand       = "#E6BE6A"   # warm sand
#c_pale_yellow= "#F2E8B6"   # pale yellow
#c_light_green= "#A6C96A"   # light green
#c_mid_green  = "#5AA14F"   # medium green
#c_dark_green = "#0B6B2E"   # dark green

# Classical region
ax.fill_between([0, n_max], Tc, Tmax, color="ivory")

# Exciton gas
ax.fill_between(n_left,
                np.minimum(T_BEC_left, Tc),
                Tc,
                color=c_pale_yellow,
                alpha=0.7)

# BEC degeneracy
ax.fill_between(n_left,
                0,
                np.minimum(T_BEC_left, Tc),
                color=c_sand,
                alpha=0.7)

# BCS region (decreasing dome)
ax.fill_between(n_bcs,
                0,
                T_BCS_boundary,
                color=c_orange,
                alpha=0.7)

# Coexistence (between dome and Td)
ax.fill_between(n_bcs,
                T_BCS_boundary,
                Tc,
                color=c_light_green,
                alpha=0.7)

# Plasma region
ax.fill_between([n_M_plot, n_max],
                0,
                Tc,
                color=c_mid_green,
                alpha=0.7)

# ============================================================
# Lines
# ============================================================
ax.plot(n_left, T_BKT_left, 'k--', linewidth=0.8)
ax.plot(n_left, T_BEC_left, 'k:', linewidth=0.8)
ax.plot(n_bcs, T_BCS_boundary, 'k:', linewidth=0.8)

ax.axhline(Tc, color="k", linestyle=":", linewidth=0.8)

#ax.plot(n_D_plot, Td_BEC, 'ko', markersize=4)
#ax.text(n_D_plot + 1, Td_BEC + 10, r"$T^\mathrm{BEC}_\mathrm{d}$", fontsize=10, ha="left", va="top")
ax.plot(n_D_plot, Td_BKT, 'ko', markersize=4)
ax.text(n_D_plot + 1, Td_BKT + 10, r"$T_\mathrm{BKT}$", fontsize=10, ha="left", va="top")
ax.plot(n_D_plot, Td_BEC, 'ko', markersize=4)
ax.text(n_D_plot + 1, Td_BEC + 10, r"$T_\mathrm{DEBG}$", fontsize=10, ha="left", va="top")

ax.vlines([n_D_plot, n_M_plot],
          ymin=0,
          ymax=Tc,
          colors="k",
          linestyles=":",
          linewidth=0.8)

# ============================================================
# Labels and annotations
# ============================================================
ax.text(n_D_plot + 0.8, 0.03*Tc, r"$n_d$", fontsize=10)
ax.text(n_M_plot + 0.8, 0.03*Tc, r"$n_M$", fontsize=10)
ax.text(n_D_plot + 12, 0.5*Tc, "Coexistence region\n (EHL/EHP + gas)", fontsize=9,ha="center", va="center")
ax.text(n_D_plot + 5, Tc + 20, r"Classical electron-hole gas", fontsize=9, ha="center", va="center")
ax.text(n_M_plot + 3, 0.5*Tc, "EHP\nor\nEHL", fontsize=9, ha="center", va="center")
ax.text(n_D_plot - 3, Tc - 220, "Exciton\ngas", fontsize=9, ha="center", va="center")
ax.text(n_D_plot - 3, 180, "DEBG", fontsize=8, ha="center", va="center")
ax.text(n_D_plot - 3, 30, "BKT", fontsize=8, ha="center", va="center")
ax.text(n_D_plot + 10, 30, "BCS", fontsize=8, ha="center", va="center")
ax.text(-1, Tc, r"$T_\mathrm{c}$", fontsize=10, ha="center", va="center")

# ============================================================
# Axes
# ============================================================
ax.set_xlabel(r"Electron-hole pair density $\ [10^{12}\ \mathrm{cm}^{-2}]$")
ax.set_ylabel("Temperature [K]")
ax.set_xlim(0, n_max)
ax.set_ylim(0, Tmax)

plt.tight_layout()
plt.savefig("phase_diagram_BE-exciton-condensates.svg",dpi=300)
plt.show()
