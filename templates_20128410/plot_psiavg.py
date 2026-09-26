import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib as mpl

# =====================================================================
# SYSTEM FONT CONFIGURATION (Times New Roman / Times)
# =====================================================================
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

# =====================================================================
# 1. DATA LOADING & NORMALIZATION
# =====================================================================
data_file = 'Psi-Average_state-1.dat'

try:
    # Loads the two columns, automatically ignoring header lines starting with '#'
    data = np.loadtxt(data_file)
    x_data = data[:, 0]
    raw_y_data = data[:, 1]
    
    # Normalize y-data so that the peak value equals 1.0
    norm_factor = np.max(raw_y_data)
    y_data = raw_y_data / norm_factor
    
    print(f"-> File '{data_file}' loaded and normalized successfully! ({len(x_data)} points)")
except FileNotFoundError:
    print(f"Error: The file '{data_file}' was not found.")
    print("Please make sure the file is in the same folder as this script.")
    sys.exit()
except Exception as e:
    print(f"Unexpected error while reading the file: {e}")
    sys.exit()

# =====================================================================
# 2. UPPER ENVELOPE EXTRACTION (Oscillation Peaks)
# =====================================================================
# 'distance' prevents picking up noise peaks too close to each other.
peak_indices, _ = find_peaks(y_data, distance=6, prominence=1e-13 / norm_factor)

# Safely ensure the absolute maximum (hole position / central peak) is included
max_idx = np.argmax(y_data)
if max_idx not in peak_indices:
    peak_indices = np.append(peak_indices, max_idx)

# Sort the indices to maintain sequential order along the X-axis
peak_indices = np.sort(peak_indices)

# Extract coordinates for the envelope points
x_envelope = x_data[peak_indices]
y_envelope = y_data[peak_indices]

print(f"-> Envelope mapped successfully using {len(x_envelope)} control points.")

# =====================================================================
# 3. GAUSSIAN MODEL DEFINITION AND CURVE FITTING
# =====================================================================
def gaussian_envelope(x, y0, A, xc, sigma):
    """Returns a Gaussian function with a vertical offset (baseline) y0."""
    return y0 + A * np.exp(-((x - xc) ** 2) / (2 * sigma ** 2))

# Automatic initial guesses based on normalized envelope data
peak_x = x_data[max_idx]
peak_y = y_data[max_idx]  # This is now exactly 1.0
y0_guess = np.min(y_envelope)
A_guess = peak_y - y0_guess
sigma_guess = 5.0  # Reasonable initial guess for the exciton spatial extent

initial_guess = [y0_guess, A_guess, peak_x, sigma_guess]

# Run the fit strictly on the normalized upper envelope points
popt, pcov = curve_fit(gaussian_envelope, x_envelope, y_envelope, p0=initial_guess)
y0_fit, A_fit, xc_fit, sigma_fit = popt
sigma_fit = abs(sigma_fit)  # Ensure positive value for physical interpretation

# =====================================================================
# 4. PHYSICAL PROPERTY CALCULATIONS (Exciton Radius)
# =====================================================================
fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma_fit
exciton_radius_sigma = sigma_fit
exciton_radius_hwhm = fwhm / 2

# Print formatted results to the terminal
print("\n" + "="*50)
print("             ENVELOPE FITTING RESULTS (NORMALIZED)")
print("="*50)
print(f" Envelope Center (xc)        : {xc_fit:.4f} Å")
print(f" Adjusted Amplitude (A)      : {A_fit:.4f}")
print(f" Baseline Level (y0)         : {y0_fit:.4f}")
print(f" Standard Deviation (sigma)  : {sigma_fit:.4f} Å")
print(f" FWHM of Envelope            : {fwhm:.4f} Å")
print("-"*50)
print(f" Exciton Radius (via sigma)  : {exciton_radius_sigma:.4f} Å  (68.3% density)")
print(f" Exciton Radius (via HWHM)   : {exciton_radius_hwhm:.4f} Å  (Half Max Width)")
print("="*50 + "\n")

# =====================================================================
# 5. PROFESSIONAL PLOT GENERATION
# =====================================================================
# Set figure size and DPI for high-quality rendering
fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=150)

# Raw background data (kept in muted gray to avoid visual clutter)
ax.plot(x_data, y_data, color='#d3d3d3', linewidth=1, label='Radially averaged', zorder=1)

# Selected envelope anchoring points (peaks)
ax.scatter(x_envelope, y_envelope, color='tab:blue', s=20, edgecolor='black',
           linewidth=0.8, label='Envelope maxima', zorder=3)

# Continuous fitted Gaussian envelope curve
x_fine = np.linspace(x_data.min(), x_data.max(), 1000)
y_fine = gaussian_envelope(x_fine, *popt)
ax.plot(x_fine, y_fine, color='#d62728', linewidth=1.0, label='Gaussian envelope Fit', zorder=4)

# ---------------------------------------------------------------------
# VISUAL SIGMA ($\sigma$) REPRESENTATION (Aligned to the Right Side)
# ---------------------------------------------------------------------
# Calculate height at 1-sigma (inflection point of the Gaussian)
y_sigma = y0_fit + A_fit * np.exp(-0.5)

# 1. Draw a subtle vertical reference line at the center (xc)
ax.vlines(xc_fit, y0_fit, peak_y-0.5, colors='black', linestyles='--', linewidth=1., zorder=2)

# 2. Draw a double-sided arrow representing the 1-sigma radius span
ax.annotate('',
            xy=(xc_fit + sigma_fit, y_sigma),
            xytext=(xc_fit, y_sigma),
            arrowprops=dict(arrowstyle='<|-|>', color='black', lw=1.2, shrinkA=0, shrinkB=0),
            zorder=5)

# 3. Draw thin vertical tick boundaries at the ends of the arrow for precision
tick_h = A_fit * 0.03  # height of the tick mark
ax.plot([xc_fit, xc_fit], [y_sigma - tick_h, y_sigma + tick_h], color='black', lw=1, zorder=5)
ax.plot([xc_fit + sigma_fit, xc_fit + sigma_fit], [y_sigma - tick_h, y_sigma + tick_h], color='black', lw=1, zorder=5)

# 4. Add the mathematical text label aligned to the right of the arrow
text_x_padding = 2.5
ax.text(xc_fit + sigma_fit + text_x_padding, y_sigma,
        rf'$\sigma = {sigma_fit:.2f}\ \mathrm{{\AA}}$',
        color='black', ha='left', va='center', fontsize=9, zorder=6)

# ---------------------------------------------------------------------

# Axis labels and aesthetic formatting
ax.set_xlabel(r'$r$ $[\mathrm{\AA{}}]$', fontsize=10, labelpad=8)
# Label updated to show normalized format: |Psi|^2 / max(|Psi|^2)
ax.set_ylabel(r'Normalized $|\Psi_S(r)|^2$', fontsize=10, labelpad=8)
ax.legend(loc='upper left', frameon=False, fancybox=False, shadow=False, fontsize=9)

# Adjust limits to pad the top/bottom nicely on a normalized 0 to 1 scale
ax.set_ylim(-0.05, 1.15)

# Clean layout spacing
plt.tight_layout()

# Save and render
plt.savefig("psi_avg.svg", dpi=300)
plt.show()
