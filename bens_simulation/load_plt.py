from scipy.fftpack import fft, fftfreq
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

