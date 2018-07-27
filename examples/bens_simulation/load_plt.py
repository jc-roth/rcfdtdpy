from scipy.fftpack import fft, fftfreq, fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt


fname = 'bounds_change_data.npz'
dat = np.load(fname)
t = dat['t']
chi_t = dat['chi_t']
Sig_back_trc = dat['Sig_back_trc']
Ref_back_trc = dat['Ref_back_trc']
Sig_trc = dat['Sig_trc']
Ref_trc = dat['Ref_trc']

"""
# CHI STUFF
fig2 = plt.figure(2)
f2_ax1 = fig2.add_subplot(1,1,1); 
f2_ax1.plot(t,chi_t)
plt.show()

# Time plot
plt.plot(t, Sig_back_trc)
plt.show()

# FFT STUFF
incf = fft(Ref_trc)
freq = fftfreq(len(t), 0.003 * (10/3))
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.plot(freq, np.absolute(incf))
ax1.plot(freq, np.angle(incf))
plt.show()

trans = fft(Sig_trc)
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.plot(freq, np.absolute(trans))
ax1.plot(freq, np.angle(trans))
plt.show()
"""

# Load in constants (MAKE SURE THAT THESE ARE UP TO DATE WITH DATA TO LOAD IN)
N = 2**16 # Taken from original.py
c0 = 1 # Taken from original.py
dt = np.diff(t)[0] # Calculate time step difference
L = 0.09

# Calculate Fourier transforms
freq = fftfreq(N, dt) # Calculate frequency axis
omega = 2*np.pi*ifftshift(freq) # Calculate angular frequency axis

# Calculate chi in frequency
chi_f = fftshift(fft(chi_t,N))

# Calculate fields in frequency
sig_spec = fft(Sig_trc-np.mean(Sig_trc),N)
ref_spec = fft(Ref_trc-np.mean(Ref_trc),N)

# Calculate spectrum in frequency
t_spec = np.divide(sig_spec, ref_spec)
phi = -np.angle(t_spec)
Abs_spec = np.absolute(t_spec)

# UNKNOWN STUFF
# Calculate n (index of refraction?!) via c0*phi/(omega*L)+1
n = np.divide(np.multiply(c0, phi), np.multiply(omega, L)) + 1

# Calculate k (wavenumber?!) via -c0/(omega*L)*log(Abs_spec*(n+1)^2/(4*n))
k = np.multiply(np.divide(-c0, np.multiply(omega, L)), np.log(np.divide(np.multiply(Abs_spec, np.square(n+1)), 4*n)))

# Perform FFT shifts
n = fftshift(n)
k = fftshift(k)

# Plot findings
plt.plot(freq,n)
plt.ylabel('n')
plt.xlabel('frequency (THz)')
plt.xlim(0.2, 2.5)
plt.show()

plt.plot(freq,k)
plt.ylabel('$\kappa$')
plt.xlabel('frequency (THz)')
plt.xlim(0.2, 2.5)
plt.show()

"""
plt.plot(freq,np.real(chi_f),freq,-np.imag(chi_f))
plt.xlabel('frequency (THz)')
plt.ylabel('$\chi(t)$ (a.u.)')
plt.xlim(0.2, 2.5)
plt.show()
"""

