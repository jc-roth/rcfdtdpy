# Imports
import numpy as np
from scipy.fftpack import fft, fftfreq
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from rcfdtd_sim import Simulation as Sim
from rcfdtd_sim import Current
from rcfdtd_sim import NumericMaterial as Mat

# ----------------------------
# Define simulation parameters
# ----------------------------
# Constants
from rcfdtd_sim.rcfdtd_sim import Mat

c0 = 3e8  # um/ps
di = 0.03e-6  # 0.03 um
dn = di / c0  # (0.03 um) / (3e8 m/s) = 0.1 fs
epsilon0 = 8.854187e-12
mu0 = np.divide(1, np.multiply(epsilon0, np.square(c0)))
# Define bounds
i0 = -1e-6  # -1 um
i1 = 2e-6  # 2 um
n0 = -0.5e-12  # -0.5 ps
n1 = 2.5e-12  # 2.5 ps
# Calculate dimensions
nlen, ilen = Sim.calc_dims(n0, n1, dn, i0, i1, di)
z, t = Sim.calc_arrays(n0, n1, dn, i0, i1, di)

# --------------------
# Define the THz pulse
# --------------------
thz_loc = -0.5e-6  # -0.5 um
thz_time = 0  # 0 fs
# Find indices
thz_loc_ind = np.argmin(np.abs(np.subtract(z, thz_loc)))
thz_time_ind = np.argmin(np.abs(np.subtract(t, thz_time)))
# Find start and end indices in time
spread = 3500
thz_time_ind_start = thz_time_ind - spread
thz_time_ind_end = thz_time_ind + spread
# Make pulse
thzpulse = np.append(np.diff(np.diff(np.exp(-((t[thz_time_ind_start:thz_time_ind_end] - thz_time) ** 2) / (8e-27)))),
                     [0, 0])
# Create Current object
thzpulse = Current(nlen, ilen, thz_time_ind_start, thz_loc_ind, thzpulse)

# ----------------------------------
# Define the dynamic material bounds
# ----------------------------------
#  Set material length
material_length = 0.2e-6  # 200 nm
# Set locations
material_start = 0
material_end = material_start + material_length
# Find indices
material_ind_start = np.argmin(np.abs(np.subtract(z, material_start)))
material_ind_end = np.argmin(np.abs(np.subtract(z, material_end)))
# Determine matrix length in indices
material_ind_len = material_ind_end - material_ind_start
# Define the high frequency permittivity
epsilon_inf = 1


# --------------------------------
# Define dynamic material behavior
# --------------------------------
# Define the ground state susceptibility
def chi_g(t):
    pass


# Define the excited state susceptibility
def chi_e(t):
    pass


# Define the fraction of excited susceptibilities
def efrac(t):
    pass


# Define the electric susceptibility chi
def chi(t):
    return efrac(t) * chi_e(t) + (1 - efrac(t)) * chi_g(t)


# Create the material
Mat(di, dn, ilen, nlen, material_ind_start, material_ind_end, chi, epsilon_inf)
