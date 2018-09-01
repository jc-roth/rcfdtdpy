# Imports
from rcfdtdpy import Simulation, Current, NumericMaterial
from examples import vis
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

# Speed of light
c0 = 3e8  # m/s
# Spatial step size
di = 0.03e-6  # 0.03 um
# Temporal step size
dn = di / c0  # (0.03 um) / (3e8 m/s) = 0.1 fs
# Permittivity of free space
epsilon0 = 8.854187e-12
# Permeability of free space
mu0 = np.divide(1, np.multiply(epsilon0, np.square(c0)))
# Define simulation bounds
i0 = -1e-6  # -1 um
i1 = 1e-6  # 1 um
n0 = -0.5e-12  # -0.5 ps
n1 = 2.5e-12  # 2.5 ps
# Calculate simulation dimensions
ilen, nlen = Simulation.calc_dims(i0, i1, di, n0, n1, dn)
# Calculate arrays that provide the spatial and temporal value of each cell
z, t = Simulation.calc_arrays(i0, i1, di, n0, n1, dn)

# Define current pulse location and time
thz_loc = -0.5e-6  # -0.5 um
thz_time = 0  # 0 fs
# Find the corresponding location and time indices
thz_loc_ind = np.argmin(np.abs(np.subtract(z, thz_loc)))
thz_time_ind = np.argmin(np.abs(np.subtract(t, thz_time)))

thzshape = np.append(np.diff(np.diff(np.exp(-(((t - thz_time)/90e-15) ** 2)))), [0, 0])

thzpulse = Current(thz_loc_ind, 0, ilen, nlen, thzshape)

# Set material length
material_length = 0.050e-6  # 50 nm
# Set locations
material_start = 0
material_end = material_start + material_length
# Find the corresponding location and time indices
material_ind_start = np.argmin(np.abs(np.subtract(z, material_start)))
material_ind_end = np.argmin(np.abs(np.subtract(z, material_end)))
# Determine matrix length in indices
material_ind_len = material_ind_end - material_ind_start

# Define constants
a = np.complex64(1e16)
gamma = np.complex64(1e12 * 2 * np.pi)

# Define electric susceptibility in time
def chi(t):
    return a*(1-np.exp(-2*gamma*t))

# Define the high frequency permittivity in time (simply a constant)
def inf_perm(t):
    return 1

# Create our material!
drude = NumericMaterial(di, dn, ilen, nlen, material_ind_start, material_ind_end, chi, inf_perm)

# Export the susceptibility of the material
drude_chi = drude.export_chi()

plt.plot(t*1e12, drude_chi)
plt.xlabel('time [ps]')
plt.ylabel('$\chi(t)$')
plt.show()

nstore = np.arange(0, int(nlen/3), 100)

s = Simulation(i0, i1, di, n0, n1, dn, epsilon0, mu0, 'absorbing', thzpulse, drude, nstore=nstore, istore=[ilen-6])
# Run simulation
s.simulate()

# View timeseries
vis.timeseries(s, z * 1e6, iunit='um')

# Export field values
hfield, efield, hfield_ref, efield_ref = s.export_ifields()

# Plot in time
plt.plot(t, np.real(efield), label='$E_{t}(t)$')
plt.plot(t, np.real(efield_ref), label='$E_{ref}(t)$')
plt.ylabel('Amplitude [?]')
plt.xlabel('time [s]')
plt.legend()
plt.show()

# Calculate time difference
dt = np.diff(t)[0] # Calculate time step difference in fs

# Calculate Fourier transforms
freq = fftfreq(nlen, dt) # in Hz
trans = fft(np.real(efield[:,0]))
ref = fft(np.real(efield_ref[:,0]))

# Remove unwanted frequencies
freq = freq[1:int(nlen/2)]
trans = trans[1:int(nlen/2)]
ref = ref[1:int(nlen/2)]

# Plot transformed fields
plt.plot(freq * 1e-12, np.abs(trans), label='$E_{t}(\omega)$')
plt.plot(freq * 1e-12, np.abs(ref), label='$E_{ref}(\omega)$')
plt.xlabel(r'frequency [THz]')
plt.xlim(0, 10)
plt.legend()
plt.show()

# Remove zero indicies from all arrays
nonzero_ind = np.nonzero(ref)
freq = freq[nonzero_ind]
ref = ref[nonzero_ind]
trans = trans[nonzero_ind]

# Calculate t
spec = np.divide(trans, ref)

# Set constants
Z0 = np.multiply(mu0, c0) # Ohms (impedance of free space)

# Calculate the angular frequency
ang_freq = 2 * np.pi * freq # THz * 2pi

# Calculate conductivity
conductivity = np.multiply(np.divide(2, Z0*material_length), np.subtract(np.divide(1, spec), 1))

# Only fit to frequencies below 14THz, as the terahertz pulse has approximately zero amplitude above 14THz
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

plt.show()