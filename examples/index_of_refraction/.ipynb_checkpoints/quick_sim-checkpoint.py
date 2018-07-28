# Imports
from rcfdtd_sim import Sim, Current, Mat, vis
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift
from matplotlib import pyplot as plt

# ================
# SETUP SIMULATION
# ================
# Constants
c0 = 1 # um/ps
di = 0.3 # 0.3 um
dn = di/c0 # (0.3 um) / (300 um/ps) = 0.001 ps = 1 fs
epsilon0 = 1
mu0 = 1
# Define bounds
i0 = -100 # -100 um
i1 = 1100 # 1100 um
n0 = -300 # (1 fs) * (-300 um) / (0.3 um/step) = (1 fs) * (-1,000 steps) = -1,000 fs = -1 ps
n1 = 5100 # (1 fs) * (5100 um) / (0.3 um/step) = (1 fs) * (17,000 steps) = 17,000 fs = 17 ps
# Calculate dimensions
nlen, ilen = Sim.calc_dims(n0, n1, dn, i0, i1, di)
print('nlen=%i, ilen=%i' % (nlen, ilen))
# Create a arrays that hold the value of the center of each cell
t = np.linspace(n0+dn/2, n1+dn/2, nlen, endpoint=False) * (10/3) # Multiply by 10/3 to get from um -> fs
z = np.linspace(i0+di/2, i1+di/2, ilen, endpoint=False)

# =============
# SETUP CURRENT
# =============
# Set current location
cp_loc_val = -50 # -250 um
cp_time_val = 0 # 0 fs
# Find current indicies
cp_loc_ind = np.argmin(np.abs(np.subtract(z, cp_loc_val)))
cp_time_ind = np.argmin(np.abs(np.subtract(t, cp_time_val)))
# Find current start and end indicies in time
spread = int(500 / 1) # (500 fs) / (1 fs/step) = 500 steps
cp_time_s = cp_time_ind - spread
cp_time_e = cp_time_ind + spread
# Make pulse
cpulse = np.append(np.diff(np.diff(np.exp(-((t[cp_time_s:cp_time_e]-cp_time_val)**2)/(2e4)))), [0,0])
# Create Current object
current = Current(nlen, ilen, cp_time_s, cp_loc_ind, cpulse)

# ==============
# SETUP MATERIAL
# ==============
# Set material length
m_len_val = 1000 # 1000 um = 1mm
# Set locations
m_s_val = 0
m_e_val = m_s_val + m_len_val
# Find indicies
m_s_ind = np.argmin(np.abs(np.subtract(z, m_s_val)))
m_e_ind = np.argmin(np.abs(np.subtract(z, m_e_val)))
# Set constants
a = np.complex64(1)
gamma = np.complex64(1e-2)
freq = np.complex64(2e-1)
# Calculate beta
ang_gamma = np.complex64(gamma * 2 * np.pi)
omega = np.complex64(freq * 2 * np.pi)
beta = np.sqrt(np.add(np.square(ang_gamma), -np.square(omega)), dtype=np.complex64)
a1 = np.complex64(a/(2*beta))
a2 = np.complex64(-a/(2*beta))
# Determine matrix length
m_len_ind = m_e_ind - m_s_ind
# Create matrices
m = np.ones((1, m_len_ind), dtype=np.complex64)
mgamma = m * ang_gamma
mbeta = m * beta
ma1 = m * a1
ma2 = m * a2
# Create material object
inf_perm = 9
material = Mat(dn, ilen, nlen, m_s_ind, inf_perm, ma1, ma2, mgamma, mbeta, storelocs=[1])
# Display constants
print('gamma=%i, beta=%i, a1=%i, a2=%i' % (gamma, beta, a1, a2))

# ==============
# RUN SIMULATION
# ==============
# Create Sim object
s = Sim(i0, i1, di, n0, n1, dn, epsilon0, mu0, 'absorbing', current, material, nstore=int(nlen/40), storelocs=[5,ilen-6])
# Run simulation
s.simulate()
# Export visualization
vis.timeseries(s, iunit='um')
# Export and save arrays
n, ls, els, erls, hls, hrls = s.export_locs()
ls_mat, chi = material.export_locs()
n = n * (10/3) # 10/3 scale factor converts from um -> fs

# ===================================
# EXTRACT SIMULATION DATA AND ANALYZE
# ===================================
# Extract incident, transmitted, and reflected fields
inc = erls[:,1]
trans = els[:,1]
refl = els[:,0] - erls[:,0]
# Calculate time difference
dn = np.diff(n)[0] # Calculate time step difference in fs
# Calculate Fourier transforms
freq = fftfreq(nlen, dn) * 1e3 # in THz (since dn=1fs, 1/dt = 1/fs = 10^15/s = 10^3*10^12/s = 10^3*THz)
incf = fft(inc)
transf = fft(trans)
# Remove unwanted frequencies
freq = freq[1:int(nlen/2)]
incf = incf[1:int(nlen/2)]
transf = transf[1:int(nlen/2)]
# Remove zero indicies from all arrays
nonzero_ind = np.nonzero(incf)
freq = freq[nonzero_ind]
incf = incf[nonzero_ind]
transf = transf[nonzero_ind]
# Calculate spectrum in frequency
spec = np.divide(transf, incf)
spec_m = np.absolute(spec)
spec_a = np.abs(np.unwrap(np.angle(spec)))

# =============================
# CALCULATE INDEX OF REFRACTION
# =============================
# Set constants (MAKE SURE THAT THESE ARE UP TO DATE WITH DATA TO LOAD IN)
c0 = 1 # 300 um/ps : Taken from original.py
m_len_adj = m_len_val/300 # 1250 um / (300 um/ps) = 125 ps / 30 : Material length (manually divided by 300 um/ps as c0 = 1)
# Calculate the angular frequency
ang_freq = 2 * np.pi * freq # THz * 2pi
# Calculate coefficients
coeff = np.divide(c0, np.multiply(ang_freq, m_len_adj))
# Calculate the real part of the index of refraction
n1 = np.multiply(coeff, spec_a) + 1
# Calculate the imaginary part of the index of refraction
kappa1 = np.multiply(-coeff, np.log(np.multiply(spec_m, np.divide(np.square(n1+1), 4*n1))))

# ====
# PLOT
# ====
# Setup figure
plt.close('all')
fig = plt.figure(figsize=(12, 8), dpi=100)
# Setup axes
ax_chi = plt.subplot2grid((5,3), (1, 0), 2, 1)
ax_freq = plt.subplot2grid((5,3), (1, 1), 2, 2)
ax_time = plt.subplot2grid((5,3), (0, 0), 1, 3)
ax_t = plt.subplot2grid((5,3), (3, 0), 1, 2, sharex = ax_freq)
ax_p = plt.subplot2grid((5,3), (4, 0), 1, 2, sharex = ax_t)
ax_n = plt.subplot2grid((5,3), (3, 2), 2, 1)
ax_k = ax_n.twinx()
# Time axis
ax_time.plot(n*1e-3, inc, label='$E_i(t)$')
ax_time.plot(n*1e-3, trans, label='$E_t(t)$')
ax_time.plot(n*1e-3, refl, label='$E_r(t)$')
ax_time.set_ylabel('amplitude [?]')
ax_time.set_xlabel('time [ps]')
ax_time.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
# Chi axis
ax_chi.plot(n, np.real(chi))
ax_chi.set_ylabel('$\chi(t)$')
ax_chi.set_xlabel('time [fs]')
ax_chi.set_xlim(-1e3,-6e2)
ax_chi.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
# Frequency axis
ax_freq.plot(freq, np.abs(incf), label='$E_i(\omega)$')
ax_freq.plot(freq, np.abs(transf), label='$E_t(\omega)$')
ax_freq.set_ylabel('amplitude [?]')
ax_freq.set_xlabel('frequency [THz]')
ax_freq.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
# Coefficient plot
ax_t.plot(freq, spec_m)
ax_t.set_ylim(0,1)
ax_t.set_ylabel(r'$E_t(\omega)/E_i(\omega)$')
ax_p.plot(freq, spec_a)
ax_p.set_ylim(0,1e3)
ax_p.set_ylabel(r'$\phi(\nu)$ [rad]')
ax_p.set_xlabel(r'frequency [THz]')
ax_p.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
# Index of refraction plot
n1_line, = ax_n.plot(ang_freq, n1, 'k-')
kappa1_line, = ax_k.plot(ang_freq, kappa1, 'k--')
ax_n.set_xlabel(r'$\omega$ [$2\pi\times$THz]')
ax_n.set_ylabel(r'$n$')
ax_k.set_ylabel(r'$\kappa$')
ax_n.set_xlim(0, 2*np.pi*1e1)
ax_n.set_ylim(2.68, 2.69)
ax_n.legend((n1_line, kappa1_line), ('$n$', '$\kappa$'), bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode='expand', borderaxespad=0)
# Final setup
ax_time.set_title(r'RC-FDTD Simulation: $\epsilon_\infty=%0.2f$, $A=%0.2f$, $\gamma=%0.2f$, $\omega=%0.2f$, L=$%0.2f$mm, THz pulse' % (inf_perm, a, gamma, omega, m_len_val))
ax_p.set_xlim(0,10)
plt.tight_layout()
# Show figure
plt.show()