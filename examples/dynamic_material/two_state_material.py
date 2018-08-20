# Imports
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.optimize import curve_fit
from pathlib import Path
from matplotlib import pyplot as plt
from rcfdtd_sim import Simulation as Sim
from rcfdtd_sim import Current
from tqdm import tqdm
from rcfdtd_sim import TwoStateMaterial as Mat
from rcfdtd_sim import vis

# ----------------------------
# Define simulation parameters
# ----------------------------
# Constants
c0 = 3e8  # um/ps
di = 0.03e-6  # 0.03 um
dn = di / c0  # (0.03 um) / (3e8 m/s) = 0.1 fs
epsilon0 = 8.854187e-12
mu0 = np.divide(1, np.multiply(epsilon0, np.square(c0)))
# Define bounds
i0 = -1e-6  # -1 um
i1 = 1e-6  # 1 um
n0 = -0.5e-12  # -0.5 ps
n1 = 2.5e-12  # 2.5 ps
# Calculate dimensions
ilen, nlen = Sim.calc_dims(i0, i1, di, n0, n1, dn)
z, t = Sim.calc_arrays(i0, i1, di, n0, n1, dn)

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
thzpulse = Current(thz_loc_ind, thz_time_ind_start, ilen, nlen, thzpulse)

# ----------------------------------
# Define the dynamic material bounds
# ----------------------------------
#  Set material length
material_length = 0.050e-6  # 50 nm
# Set locations
material_start = 0
material_end = material_start + material_length
# Find indices
material_ind_start = np.argmin(np.abs(np.subtract(z, material_start)))
material_ind_end = np.argmin(np.abs(np.subtract(z, material_end)))
# Determine matrix length in indices
material_ind_len = material_ind_end - material_ind_start


# ----------------------------------------------
# Define numeric material behavior (Drude metal)
# ----------------------------------------------
#  Set constants
a = np.complex64(1e16)
gamma = np.complex64(1e12 * 2 * np.pi)
inf_perm = np.complex64(1e0)
alpha = 5 / 1e9 # 5 / nm
Gamma = 100e-15 # 100 fs
t_diff = 0
tau = 30e-15 # 30 fs
b = 0

# Create matrices
m = np.ones((1, material_ind_len), dtype=np.complex64)
mgamma = m * gamma
mbeta = m * gamma
ma1 = m * a
ma2 = -1 * m * a

from scipy.integrate import quad

def integrand(tp):
    p1 = np.exp(-np.square(np.divide(t_diff - tp, Gamma)))
    p2 = np.add(np.exp(-np.divide(tp, tau)), b)
    return np.multiply(p1, p2)

t_trim_ind = np.argmin(np.abs(np.subtract(t, 0)))
t_trim = t[t_trim_ind:]

# FIGURE OUT HOW TO NORMALIZE THIS!!!
ints = np.zeros(len(t_trim))
for i in tqdm(range(len(t_trim))):
    area, err = quad(integrand, -10e-12, t_trim[i], limit=500)
    ints[i] = area

plt.plot(t_trim * 1e15, ints)
plt.xlabel('t [fs]')
plt.show()

#drude_material = Mat(di, dn, ilen, nlen, material_ind_start, ma1, ma2, mbeta, mgamma, ma1, ma2, mbeta, mgamma, ma1, ma2, mbeta, mgamma, alpha, Gamma, t_diff, tau, b, inf_perm)

"""
# --------------------------
# Run or load the simulation
# --------------------------
# Run simulation if simulation save doesn't exist
sim_file = Path('numeric_material.npz')
if sim_file.is_file():
    # Load results
    dat = np.load('numeric_material.npz')
    t = dat['t']
    els = dat['els']
    erls = dat['erls']
    hls = dat['hls']
    hrls = dat['hrls']
    chi = dat['chi']
else:
    # Create the material
    drude_material = Mat(di, dn, ilen, nlen, material_ind_start, material_ind_end, chi, inf_perm, tqdmarg={'desc':'Calculating chi^m'})
    # Create Sim object
    tqdmarg = {'desc': 'Executing simulation'}
    s = Sim(i0, i1, di, n0, n1, dn, epsilon0, mu0, 'absorbing', thzpulse, drude_material, nstore=np.arange(0, nlen, 50), istore=[5,ilen-6])
    # Run simulation
    s.simulate(tqdmarg)
    # Export visualization
    vis.timeseries(s, z, iunit='um')
    # Export and save arrays
    hls, els, hrls, erls = s.export_ifields()
    chi = drude_material.export_chi()
    np.savez('numeric_material.npz', t=t, els=els, erls=erls, hls=hls, hrls=hrls, chi=chi)

# ---
# Chi
# ---
plt.plot(t*1e12, chi)
plt.show()

# ------
# Fields
# ------
# Extract incident, transmitted, and reflected fields
inc = erls[:,1]
trans = els[:,1]
refl = els[:,0] - erls[:,0]

# Plot
plt.clf()
plt.plot(t, np.real(inc), label='$E_i(t)$')
plt.plot(t, np.real(trans), label='$E_t(t)$')
plt.plot(t, np.real(refl), label='$E_r(t)$')
plt.ylabel('Amplitude [?]')
plt.xlabel('time [s]')
plt.legend()
plt.show()


# -----------
# Frequencies
# -----------
# Calculate time difference
dt = np.diff(t)[0] # Calculate time step difference in fs

# Calculate Fourier transforms
freq = fftfreq(nlen, dt) # in THz (since [dt]=[fs], 1/[dt] = 1/[fs] = 10^15/[s] = 10^3*10^12/[s] = 10^4*[THz])
incf = fft(inc)
transf = fft(trans)
reflf = fft(refl)

# Removeunwanted frequencies
freq = freq[1:int(nlen/2)]
incf = incf[1:int(nlen/2)]
transf = transf[1:int(nlen/2)]
reflf = reflf[1:int(nlen/2)]

# Plot transformed fields
plt.plot(freq * 1e-12, np.abs(incf), label='$E_i(\omega)$')
plt.plot(freq * 1e-12, np.abs(transf), label='$E_t(\omega)$')
plt.plot(freq * 1e-12, np.abs(reflf), label='$E_r(\omega)$')
plt.xlabel(r'frequency [THz]')
plt.xlim(0, 10)
plt.legend()
plt.show()

# ------------
# Transmission
# ------------
# Remove zero indicies from all arrays
nonzero_ind = np.nonzero(incf)
freq = freq[nonzero_ind]
incf = incf[nonzero_ind]
transf = transf[nonzero_ind]
reflf = reflf[nonzero_ind]

# Calculate spectrum in frequency
spec = np.divide(transf, incf)

# Remove zero indicies from all arrays
nonzero_ind = np.nonzero(spec)
freq = freq[nonzero_ind]
incf = incf[nonzero_ind]
transf = transf[nonzero_ind]
spec = spec[nonzero_ind]

# Extract phase and magnitude
spec_m = np.absolute(spec)
spec_a = np.abs(np.unwrap(np.angle(spec)))

# Plot
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, dpi=100)
ax0.plot(freq * 1e-12, spec_m)
ax1.plot(freq * 1e-12, spec_a)
ax1.set_xlim(0, 15)
ax0.set_ylim(0, 2)
ax1.set_ylim(0, 0.5)
ax0.set_ylabel(r'$\left|E_t(\omega)/E_i(\omega)\right|$')
ax1.set_ylabel(r'$\phi$ [radians]')
ax1.set_xlabel(r'frequency [THz]')
plt.show()

# ------------
# Conductivity
# ------------
# Set constants
Z0 = np.multiply(mu0, c0) # Ohms (impedance of free space)

# Calculate the angular frequency
ang_freq = 2 * np.pi * freq # THz * 2pi

# Calculate conductivity
conductivity = np.multiply(np.divide(2, Z0*material_length), np.subtract(np.divide(1, spec), 1))

# Calculate index of refraction
#n_complex = np.sqrt(inf_perm + np.divide(np.multiply(1j, conductivity), np.multiply(ang_freq, epsilon0)))

# Calculate the imaginary part of the index of refraction
#n1 = np.real(n_complex)
#kappa1 = np.imag(n_complex)

# Setup plot
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, dpi=100)
ax0.set_ylabel(r'$\sigma_1$')
ax1.set_ylabel(r'$\sigma_2$')
ax1.set_xlabel(r'$\omega$ [THz]')
ax1.set_xlim(0, 15)
ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax0.set_ylim(0, 1.1e5)
ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax1.set_ylim(-6e4, 0)

# Plot conductivity
ax0.plot(freq * 1e-12, np.real(conductivity), 'b-')
ax1.plot(freq * 1e-12, np.imag(conductivity), 'b-')

# --------
# Plotting
# --------
# Find max frequency
freq_max = np.argmin(np.abs(np.subtract(14e12, freq)))

# Define fit functions
def cond_real(omega, sigma0, tau):
    return sigma0/(1+(tau*omega)**2)

def cond_imag(omega, sigma0, tau):
    return (-omega*tau*sigma0)/(1+(tau*omega)**2)

# Take real and imaginary parts
cfreq = freq[:freq_max]
creal = np.real(conductivity)[:freq_max]
cimag = np.imag(conductivity)[:freq_max]

# Run curve fit
popt_real, pcov_real = curve_fit(cond_real, cfreq, creal, p0=[1e5, 0.4e-12])
popt_imag, pcov_imag = curve_fit(cond_imag, cfreq, cimag, p0=[1e5, 0.2e-12])

fit_real = cond_real(freq, *popt_real)
fit_imag = cond_imag(freq, *popt_imag)

# Setup plot
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, dpi=100)
ax0.set_ylabel(r'$\sigma_1$', fontsize=15)
ax1.set_ylabel(r'$\sigma_2$', fontsize=15)
ax1.set_xlabel(r'$\omega$ [THz]', fontsize=15)
ax0.set_title(r'Drude Model (numeric)', fontsize=15)
ax1.set_xlim(0, 15)
ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax0.tick_params(labelsize=15)
ax0.set_ylim(0, 1.1e5)
ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax1.tick_params(labelsize=15)
ax1.set_ylim(-6e4, 0)

# Plot simulated conductivity
ax0.plot(freq*1e-12, np.real(conductivity), 'b-', label='simulation')
ax1.plot(freq*1e-12, np.imag(conductivity), 'b-', label='simulation')

# Plot analytic conductivity
ax0.plot(freq*1e-12, fit_real, 'r--', label='analytic')
ax1.plot(freq*1e-12, fit_imag, 'r--', label='analytic')

ax0.legend()
ax1.legend()

plt.tight_layout()

plt.savefig('numeric_material.pdf', format='pdf')
plt.show()
"""