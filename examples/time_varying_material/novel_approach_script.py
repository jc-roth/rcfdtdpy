# Imports
from rcfdtd_sim import Sim, Current, Mat, vis
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.special import erf
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

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
nlen, ilen = Sim.calc_dims(n0, n1, dn, i0, i1, di)
# Create a arrays that hold the value of the center of each cell
t = np.linspace(n0+dn/2, n1+dn/2, nlen, endpoint=False)
z = np.linspace(i0+di/2, i1+di/2, ilen, endpoint=False)
# Print simulation bounds
print('nlen=%i, ilen=%i' % (nlen, ilen))

cp_loc_val = -0.5e-6 # -0.5 um
cp_time_val = 0 # 0 fs

# Find indicies
cp_loc_ind = np.argmin(np.abs(np.subtract(z, cp_loc_val)))
cp_time_ind = np.argmin(np.abs(np.subtract(t, cp_time_val)))
# Find start and end indicies in time
spread = 3500
cp_time_s = cp_time_ind - spread
cp_time_e = cp_time_ind + spread

# Make pulse
cpulse = np.append(np.diff(np.diff(np.exp(-((t[cp_time_s:cp_time_e]-cp_time_val)**2)/(8e-27)))), [0,0])
# Create Current object
current = Current(nlen, ilen, cp_time_s, cp_loc_ind, cpulse)

# Set material length
m_len = 0.2e-6 # 200 nm
# Set locations
m_s_val = 0
m_e_val = m_s_val + m_len

m_s_ind = np.argmin(np.abs(np.subtract(z, m_s_val)))
m_e_ind = np.argmin(np.abs(np.subtract(z, m_e_val)))
# Determine matrix length in indicies
mlen = m_e_ind - m_s_ind

# Set constants
g_a1 = np.complex64(0)
g_a2 = np.complex64(1e16)
g_gamma = np.complex64(1e12)
g_freq = np.complex64(0)
# Calculate beta
g_ang_gamma = np.complex64(g_gamma * 2 * np.pi)
g_omega = np.complex64(g_freq * 2 * np.pi)
g_beta = np.sqrt(np.add(np.square(g_ang_gamma), -np.square(g_omega)), dtype=np.complex64)
# Create matrices
g_m = np.ones((1, mlen), dtype=np.complex64)
g_mgamma = g_m * g_ang_gamma
g_mbeta = g_m * g_beta
g_ma1 = g_m * g_a1
g_ma2 = g_m * g_a2

# Set constants
e_a1 = np.complex64(1e16)
e_a2 = np.complex64(-1e16)
e_gamma = np.complex64(1e12)
e_freq = np.complex64(0)
# Calculate beta
e_ang_gamma = np.complex64(e_gamma * 2 * np.pi)
e_omega = np.complex64(e_freq * 2 * np.pi)
e_beta = np.sqrt(np.add(np.square(e_ang_gamma), -np.square(e_omega)), dtype=np.complex64)
# Create matrices
e_m = np.ones((1, mlen), dtype=np.complex64)
e_mgamma = e_m * e_ang_gamma
e_mbeta = e_m * e_beta
e_ma1 = e_m * e_a1
e_ma2 = e_m * e_a2

pulse_delays = np.arange(-240e-15, 450e-15, 10e-15) # -500 fs to 500 fs at intervals of 100 fs

def e_osc_frac(pulse_delay):
    # Set constants
    pulse_width = 150e-15 # 100 fs pulse width
    pulse_scaling = 30e11 # Sets the amplitude of the optical pulse amplitude
    state_decay_const = 0.8e-12 # 0.47 ps pulse decay time
    absorption_const = 2e-1 # 400 cm^-1 absorption coefficient
    material_indicies = np.arange(0, mlen, 1)
    # Calculate excited oscillator fraction
    e_frac_exp1 = np.exp((pulse_width**2)/(4 * (state_decay_const)**2) - pulse_delay/state_decay_const)
    e_frac_exp2 = np.exp(-absorption_const*material_indicies)
    e_frac = pulse_scaling*pulse_width*np.sqrt(np.pi)*(e_frac_exp1)*e_frac_exp2
    return e_frac

f = np.zeros((len(pulse_delays), mlen))

for i in range(len(pulse_delays)):
    f[i] = e_osc_frac(pulse_delays[i])
    
plt.plot(pulse_delays * 1e12, f)
plt.xlabel('pulse delay [ps]')
plt.ylabel('$f_e$')
plt.title('$f_e$ as a function of pulse delay at each material index')
plt.show()

#Create Sim object
sim_name = 'novel_approach_sim7.npz'
if Path(sim_name).is_file():
    # Load results
    dat = np.load(sim_name)
    n = dat['n']
    pulse_delays = dat['pulse_delays']
    inc_ars = dat['inc_ars']
    trans_ars = dat['trans_ars']
    refl_ars = dat['refl_ars']
    chi_ars = dat['chi_ars']
else:
    # Create arrays to hold simulation values, each new simulation will contribute to a row. The zero-valued row initialized here is removed later.
    inc_ars = np.zeros((1, nlen))
    trans_ars = np.zeros((1, nlen))
    refl_ars = np.zeros((1, nlen))
    chi_ars = np.zeros((1, nlen))
    # Loop through each transition index, simulating at each
    for i in range(len(pulse_delays)):
        # Get the current pulse delay
        pulse_delay = pulse_delays[i]
        # Generate the oscillator fractions
        e_frac = e_osc_frac(pulse_delay)
        g_frac = 1 - e_frac
        # Scale coefficients by the excited and ground fractions
        e_ma1_scaled = e_frac * e_ma1
        e_ma2_scaled = e_frac * e_ma2
        g_ma1_scaled = g_frac * g_ma1
        g_ma2_scaled = g_frac * g_ma2
        # Combine oscillators
        ma1 = np.vstack((e_ma1_scaled, g_ma1_scaled))
        ma2 = np.vstack((e_ma2_scaled, g_ma2_scaled))
        mgamma = np.vstack((e_mgamma, g_mgamma))
        mbeta = np.vstack((e_mbeta, g_mbeta))
        # Create material
        inf_perm = np.complex64(1e0)
        two_state_mat = Mat(dn, ilen, nlen, m_s_ind, inf_perm, ma1, ma2, mgamma, mbeta, storelocs=[1])
        # Create and run simulation
        s = Sim(i0, i1, di, n0, n1, dn, epsilon0, mu0, 'absorbing', current, two_state_mat, nstore=int(nlen/50), storelocs=[5,ilen-6])
        tqdmarg = {'desc': ('Working on transition ' + str(i+1) + '/' + str(len(pulse_delays))), 'leave': False}
        s.simulate(tqdmarg)
        # Extract incident, transmitted, and reflected fields
        n, ls, els, erls, hls, hrls = s.export_locs()
        inc = erls[:,1]
        trans = els[:,1]
        refl = els[:,0] - erls[:,0]
        # Extract chi values
        ls, chi = two_state_mat.export_locs()
        # Reshape chi values
        chi = np.reshape(chi, (1, nlen))
        # Put results into array
        inc_ars = np.vstack((inc_ars, inc))
        trans_ars = np.vstack((trans_ars, trans))
        refl_ars = np.vstack((refl_ars, refl))
        chi_ars = np.vstack((chi_ars, chi))
    # Remove the first row of each array, which is the zero-valued row initialized earlier
    inc_ars = inc_ars[1:,:]
    trans_ars = trans_ars[1:,:]
    refl_ars = refl_ars[1:,:]
    chi_ars = chi_ars[1:,:]
    # Save data
    np.savez(sim_name, n=n, pulse_delays=pulse_delays, inc_ars=inc_ars, trans_ars=trans_ars, refl_ars=refl_ars, chi_ars=chi_ars)

plt.plot(n*1e12, np.real(chi_ars.T))
plt.show()

# Remove last row and column of the trans_ars array (while also taking the transpose and real part) so that it fits in our grid
trans_ars_to_plot = np.real(trans_ars.T)[:-1, :-1]

# Make grid
ddn = np.diff(pulse_delays)[0]
time_grid, dtime_grid = np.mgrid[slice(n[0], n[-1] + dn, dn),
                slice(pulse_delays[0],pulse_delays[-1] + ddn, ddn)]

# Setup colorbar
cmap = plt.get_cmap('PiYG')
levels = MaxNLocator(nbins=500).tick_values(trans_ars_to_plot.min(), trans_ars_to_plot.max())
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# Plot and add colorbar
plt.pcolormesh(dtime_grid*1e15, time_grid*1e12, trans_ars_to_plot, cmap=cmap, norm=norm)
plt.colorbar()

# Label plot
plt.ylabel('$t$ [ps]')
plt.xlabel('$\Delta t$ [fs]')
plt.gcf().set_dpi(100)
plt.show()

# Do some other things
index_to_extract = np.argmin(np.abs(np.subtract(n, 1.55e-15)))
plt.plot(pulse_delays, np.real(trans_ars.T)[index_to_extract])
plt.ylabel('$E_t(t)$')
plt.xlabel('$\Delta t$')
plt.show()
