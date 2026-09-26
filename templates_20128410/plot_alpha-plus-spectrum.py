import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

# -------------------------------------------------
# Style
# -------------------------------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{mathptmx}"
})

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5
})

# -------------------------------------------------
# Import optical data
# -------------------------------------------------
data = np.genfromtxt('o-BSE.alpha_q1_haydock_bse', usecols=(0, 1, 2, 3))
data_rpa = np.genfromtxt('o-RPA.alpha_q1_slepc_bse', usecols=(0, 1, 2, 3))
energy = data[:, 0]
alpha_Im_BSE = data[:, 1]
alpha_Im_IPA = data_rpa[:, 1]

Lz = 28.34590          # bohr radii
E_qp_g0w0 =  2.189691  # fundamental gap 1.952281 eV

# Physical constants
hbar = 6.62607e-16     # eV.s
a0 = 5.291772109e-11  # m
c_a0 = 2999792458 / a0  # a0/s

# -------------------------------------------------
# Absorbance from 2D polarizability
# -------------------------------------------------
absorbance_BSE_percent = 100 * ((4 * np.pi / (hbar * c_a0)) * energy * alpha_Im_BSE)
absorbance_IPA_percent = 100 * ((4 * np.pi / (hbar * c_a0)) * energy * alpha_Im_IPA)

# -------------------------------------------------
# Read exciton spectrum
# -------------------------------------------------
# Expected columns: Energy(eV) | Oscillator strength
exc = np.genfromtxt('o-BSE.exc_qpt1_E_sorted')

exc_energy = exc[:, 0]
strengths = exc[:, 1]

# Bright if >= 10% of max oscillator strength
max_strength = np.max(strengths)
threshold_bright = 0.10 * max_strength
is_bright = strengths >= threshold_bright

# Colors
color_bright = "#8b0000"   # "#FFD200" yellow
color_dark = "#c5d0d5"     # "#8B0000" dark red

# -------------------------------------------------
# Figure and independent axes
# -------------------------------------------------
fig = plt.figure(figsize=(3.94, 2.8), constrained_layout=True)

gs = fig.add_gridspec(2, 1, height_ratios=[5, 0.5], hspace=0.1)

ax = fig.add_subplot(gs[0])
ax_exc = fig.add_subplot(gs[1])

# --- Main absorbance plot ---
#Previous colors "#c3ab5f" and "#9eb0b9"
ax.plot(energy, absorbance_BSE_percent,
        label=r'$G_0W_0+$BSE', linestyle='-', color="chocolate", linewidth=0.8,zorder=1)
ax.fill_between(energy, absorbance_BSE_percent, color="chocolate", alpha=0.8,zorder=1)
#Old GW+BSE line color #9eb0b9

ax.plot(energy, absorbance_IPA_percent,
        label=r'IQP-RPA', linestyle='-', color="steelblue", linewidth=0.8,zorder=-1)
ax.fill_between(energy, absorbance_IPA_percent, color="steelblue", alpha=0.8, zorder=-1)

ax.axvline(x=E_qp_g0w0, linestyle='--', color='k', linewidth=0.65)
ax.annotate(r'$E_g^{\mathrm{d}}$',
            xy=(E_qp_g0w0, 2.6),
            xytext=(E_qp_g0w0 + 0.10, 1.8),
            textcoords='data',
            fontsize=9,
            rotation=0,
            va='top',
            ha='left',
            )
            
ax.set_ylabel(r"$A(\omega)$ [$\%$]")
ax.tick_params(axis='both', which='both',
               direction='in', bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=False)  # hide x labels only here
ax.legend(loc='upper left', fontsize=9, frameon=False)

ax.set_xlim(0, 4)
ax.set_ylim(0, 3)
ax.locator_params(axis='y', nbins=4)

# --- Visible light background ---
def add_vis_spectrum(ax,zorder=0):
    """Adds visible light background."""
    vis_min = 1240 / 700  # 1.77 eV
    vis_max = 1240 / 400  # 3.10 eV
    colors = [(1, 0, 0), (1, 0.6, 0), (1, 1, 0), (0, 0.5, 0), (0, 0, 1), (0.5, 0, 0.8)]
    cmap = mcolors.LinearSegmentedColormap.from_list("visible", colors)
    norm = mcolors.Normalize(vmin=vis_min, vmax=vis_max)
    for i in np.linspace(vis_min, vis_max, 256):
        ax.axvspan(i, i + (vis_max - vis_min)/256, color=cmap(norm(i)), alpha=0.05, zorder=zorder)
add_vis_spectrum(ax, zorder=-10)

# --- Exciton spectrum subplot ---
ax_exc.set_facecolor('white')

for i, e in enumerate(exc_energy):
    if is_bright[i]:
        ax_exc.vlines(e, 0, 1, color=color_bright, linewidth=1.0, zorder = 1)
    else:
        ax_exc.vlines(e, 0, 1, color=color_dark, alpha=1.0, linewidth=0.8, zorder = -1)

ax_exc.axvline(x=E_qp_g0w0, color='k', linestyle='--', linewidth=0.65)

ax_exc.set_ylim(0, 1)
ax_exc.set_xlim(0, 4)
ax_exc.set_yticks([])
ax_exc.set_xlabel(r"Photon energy [eV]", fontsize=10)
xticks = np.linspace(0, 4, 5)
ax_exc.set_xticks(xticks)


ax_exc.tick_params(axis='x', which='both',
                   labelsize=10,
                   direction='out',
                   bottom=True, top=True,
                   labelbottom=True,
                   colors='k')

for spine in ax_exc.spines.values():
    spine.set_color('k')


# -------------------------------------------------
# Save
# -------------------------------------------------
#plt.tight_layout()
fig.savefig('alpha_bse_with_exciton_spectrum.jpg', dpi=800)
plt.show()

