from scipy.fftpack import fft, fftfreq, fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt

# Load data
fname = 'ben_sim.npz'
dat = np.load(fname)
t = dat['t']
z_pos = dat['z_pos']
chi_t = dat['chi_t']
Sig_trc = dat['Sig_trc']
Ref_trc = dat['Ref_trc']

# Plot chi
plt.plot(chi_t)
plt.show()

# Find length
t_len = len(t)

# Extract incident, transmitted, and reflected fields
inc = Ref_trc
trans = Sig_trc

# Plot
plt.plot(t, np.real(inc), label='$E_i$')
plt.plot(t, np.real(trans), label='$E_t$')
plt.ylabel('Amplitude [?]')
plt.xlabel('time [fs]')
plt.legend()
plt.show()

# Calculate time difference
dt = np.diff(t)[0] # Calculate time step difference in fs

# Calculate Fourier transforms
freq = fftfreq(t_len, dt) * 1e3 # in THz (since [dn]=[fs], 1/[dn] = 1/[fs] = 10^15/[s] = 10^3*10^12/[s] = 10^4*[THz])
incf = fft(inc)
transf = fft(trans)

# Removeunwanted frequencies
freq = freq[1:int(t_len/2)]
incf = incf[1:int(t_len/2)]
transf = transf[1:int(t_len/2)]

# Remove zero indicies from all arrays
nonzero_ind = np.nonzero(incf)
freq = freq[nonzero_ind]
incf = incf[nonzero_ind]
transf = transf[nonzero_ind]

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
ax0.plot(freq, spec_m)
ax1.plot(freq, spec_a)
#ax0.set_ylim(0.0, 1.5)
#ax1.set_ylim(-1e-1, 1e-1)
#ax1.set_xlim(0, 1e1)
ax0.set_ylabel(r't')
ax1.set_ylabel(r'phase')
ax1.set_xlabel(r'frequency [THz]')
plt.show()


# Set constants
L = 0.010 * 1e-6
inf_perm = 1
Z0 = 376.73 # Ohms (impedance of free space)
permittivity_free_space = 8.854187817e-12

# Calculate the angular frequency
ang_freq = 2 * np.pi * freq # THz * 2pi

# Calculate conductivity
conductivity = np.multiply(np.divide(2, Z0*L), np.subtract(np.divide(1, spec), 1))

# Calculate index of refraction
n_complex = np.sqrt(inf_perm + np.divide(np.multiply(1j, conductivity), np.multiply(ang_freq, permittivity_free_space)))

# Calculate the imaginary part of the index of refraction
n1 = np.real(n_complex)
kappa1 = np.imag(n_complex)

# Calculate sigma1 and sigma2
sigma1 = np.real(conductivity)
sigma2 = np.imag(conductivity)

f_max = np.argmin(np.abs(np.subtract(freq, 8000)))

r_max = np.max(sigma1[0:f_max])
r_min = np.min(sigma1[0:f_max])

i_max = np.max(sigma2[0:f_max])
i_min = np.min(sigma2[0:f_max])

# Setup plot
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, dpi=100)
ax0.set_xlabel(r'$\omega$ [THz]')
ax0.set_ylabel(r'$\mathcal{Re}(\sigma)$')
ax1.set_ylabel(r'$\mathcal{Im}(\sigma)$')

# Plot conductivity
ax0.plot(freq, np.real(conductivity), 'b-')
ax1.plot(freq, np.imag(conductivity), 'b-')


# Set limits
ax1.set_xlim(0, 10000)
ax0.set_ylim(r_min, r_max)
ax1.set_ylim(i_min, i_max)

plt.show()