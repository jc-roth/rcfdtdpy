# Imports
from rcfdtd_sim import Simulation as Sim
from rcfdtd_sim import Current as Current
from rcfdtd_sim import NumericMaterial as Mat
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.special import erf
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# ----------------
# Setup simulation
# ----------------
# Constants
c0 = 3e8 # um/ps
di = 0.03e-6 # 0.03 um
dn = di/c0 # (0.03 um) / (3e8 m/s) = 0.1 fs
epsilon0 = 8.854187e-12
mu0 = np.divide(1, np.multiply(epsilon0, np.square(c0)))
# Define bounds
i0 = -1e-6 # -1 um
i1 = 2e-6 # 2 um
n0 = -0.5e-12 # -0.5 ps
n1 = 2.5e-12 # 2.5 ps
# Calculate dimensions
ilen, nlen = Sim.calc_dims(i0, i1, di, n0, n1, dn)
z, t = Sim.calc_arrays(i0, i1, di, n0, n1, dn)
# Print simulation bounds
print('nlen=%i, ilen=%i' % (nlen, ilen))

# -------------
# Setup current
# -------------
cp_z_val = -0.5e-6 # -0.5 um
cp_n_val = 0 # 0 fs
# Find indicies
cp_z_ind = np.argmin(np.abs(np.subtract(z, cp_z_val)))
cp_n_ind = np.argmin(np.abs(np.subtract(t, cp_n_val)))
# Find start and end indicies in time
spread = 3500
cp_n_start = cp_n_val - spread
cp_n_end = cp_n_val + spread
# Make pulse
cpulse = np.append(np.diff(np.diff(np.exp(-((t[cp_n_start:cp_n_end]-cp_n_val)**2)/(8e-27)))), [0,0])
# Create Current object
current = Current(nlen, ilen, cp_n_start, cp_z_ind, cpulse)

# -------------------------
# Setup material dimensions
# -------------------------
# Set material length
m_len_val = 0.05e-6 # 50 nm
# Set locations
m_z_start_val = 0
m_z_end_val = m_z_start_val + m_len_val
# Calculate indices
m_z_start_ind = np.argmin(np.abs(np.subtract(z, m_z_start_val)))
m_z_end_ind = np.argmin(np.abs(np.subtract(z, m_z_end_val)))
# Determine matrix length in indicies
m_len_ind = m_z_end_ind - m_z_start_ind

# -----------------------------
# Setup material susceptibility
# -----------------------------
# Define ground state chi
a_g = np.complex64(1e16)
gamma_g = np.complex64(1e12 * 2 * np.pi)
def chi_g(t):
    return a_g*(1-np.exp(-2*gamma_g*t))

# Define excited state chi
a_e = np.complex64(1e16)
gamma_e = np.complex64(1e12 * 2 * np.pi)
def chi_e(t):
    return a_e*(1-np.exp(-2*gamma_e*t))

# Define the fraction of excited oscillators
def f_e(t)
    return 0

# Define the susceptibility
def chi(t):
    return (f_e(t) * chi_e(t)) + ((1-f_e(t)) * chi_g(t))

# Define the infinite frequency permittivity
def inf_perm(t):
    return 1


Material(di, dn, ilen, nlen, m_z_start_ind, m_z_end_ind, chi, inf_perm, tqdmarg={})