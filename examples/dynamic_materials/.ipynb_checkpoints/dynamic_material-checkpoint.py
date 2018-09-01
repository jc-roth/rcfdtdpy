# Imports
from rcfdtdpy import Simulation as Sim
from rcfdtdpy import Current as Current
from rcfdtdpy import NumericMaterial as Mat
import numpy as np
from scipy.special import erf
from matplotlib import pyplot as plt
from pathlib import Path
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# -----------------
# Script parameters
# -----------------
development = True

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
i1 = 1e-6 # 2 um
n0 = -0.5e-12 # -0.5 ps
n1 = 2.5e-12 # 2.5 ps
# Calculate dimensions
ilen, nlen = Sim.calc_dims(i0, i1, di, n0, n1, dn)
z, t = Sim.calc_arrays(i0, i1, di, n0, n1, dn)
# Print simulation bounds
if development:
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
cp_n_start = cp_n_ind - spread
cp_n_end = cp_n_ind + spread
# Make pulse
cpulse = np.append(np.diff(np.diff(np.exp(-((t[cp_n_start:cp_n_end]-cp_n_val)**2)/(8e-27)))), [0,0])
# Create Current object
thzpulse = Current(cp_z_ind, cp_n_start, ilen, nlen, cpulse)

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

# ---------------------------
# Run pre-simulation analysis
# ---------------------------
# Run simulation if simulation save doesn't exist
sim_file = 'dynamic_material_pre_analysis.npz'
if Path(sim_file).is_file():
    # Load results
    dat = np.load(sim_file)
    els = dat['els']
else:
    # Create Sim object, observe the field at the material start index
    s = Sim(i0, i1, di, n0, n1, dn, epsilon0, mu0, 'absorbing', thzpulse, istore=[m_z_start_ind])
    # Run simulation
    s.simulate(tqdmarg={'desc': 'Executing pre-simulation analysis', 'leave': True})
    hls, els, hrls, erls = s.export_ifields()
    np.savez(sim_file, els=els)

# Determine the temporal index at which the thz pulse is incident on the material
thz_incident_n_ind = np.argmax(np.real(els))
thz_incident_n_val = t[thz_incident_n_ind]

# Plot
if development:
    # Plot
    plt.plot(t * 1e12, np.real(els))
    plt.axvline(t[thz_incident_n_ind], color='k', linestyle='--')
    plt.xlabel('time [ps]')
    plt.ylabel('$E(t,z=m_0)$')
    plt.show()
    # Print time value
    print('t[thz_incident_n_ind]=thz_incident_n_val=%E' % t[thz_incident_n_ind])

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
gamma_e = np.complex64(1e10 * 2 * np.pi)
def chi_e(t):
    return a_e*(1-np.exp(-2*gamma_e*t))

# Define the fraction of excited oscillators
vis_fwhm = 100e-15 # 100 fs
tau_decay = 1e-12 # 1 ps

def f_e(t, t_diff):
    # If statement acts as Heaviside function
    if(t >= thz_incident_n_val):
        return 0.5*(erf((t-thz_incident_n_val-t_diff)/vis_fwhm)+1)*np.exp(-(t-thz_incident_n_val-t_diff)/tau_decay)
    else:
        return 0

# Define the susceptibility
def chi(t, t_diff):
    return (f_e(t, t_diff) * chi_e(t)) + ((1-f_e(t, t_diff)) * chi_g(t))

# Define the infinite frequency permittivity
def inf_perm(t):
    return 1

# Define the t_diffs to test
t_diffs = np.arange(-1000e-15, 1000e-15, 50e-15)

# --------------------------
# Run or load the simulation
# --------------------------
# Run simulation if simulation save doesn't exist
sim_file = Path('dynamic_material_simulation.npz')
if sim_file.is_file():
    # Load results
    dat = np.load('dynamic_material_simulation.npz')
    t = dat['t']
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
    for i in range(len(t_diffs)):
        # Wrap chi function
        def chi_wrapped(t):
            return chi(t, t_diffs[i])
        # Create the material
        tqdmarg = {'desc': ('Calculating chi^m ' + str(i+1) + '/' + str(len(t_diffs))), 'leave': False}
        overlap_mat = Mat(di, dn, ilen, nlen, m_z_start_ind, m_z_end_ind, chi_wrapped, inf_perm, tqdmarg=tqdmarg)
        # Create Sim object
        s = Sim(i0, i1, di, n0, n1, dn, epsilon0, mu0, 'absorbing', thzpulse, overlap_mat, nstore=np.arange(0, nlen, 50), istore=[5,ilen-6])
        tqdmarg = {'desc': ('RC-FDTD simulation ' + str(i+1) + '/' + str(len(t_diffs))), 'leave': False}
        # Run simulation
        s.simulate(tqdmarg)
        # Extract and save the incident, transmitted, and reflected fields
        hls, els, hrls, erls = s.export_ifields()
        inc = erls[:,1]
        trans = els[:,1]
        refl = els[:,0] - erls[:,0]
        # Extract and reshape chi values
        chi_val = overlap_mat.export_chi()
        chi_val = np.reshape(chi_val, (1, nlen))
        # Put results into array
        inc_ars = np.vstack((inc_ars, inc))
        trans_ars = np.vstack((trans_ars, trans))
        refl_ars = np.vstack((refl_ars, refl))
        chi_ars = np.vstack((chi_ars, chi_val))
    # Remove the first row of each array, which is the zero-valued row initialized earlier
    inc_ars = inc_ars[1:,:]
    trans_ars = trans_ars[1:,:]
    refl_ars = refl_ars[1:,:]
    chi_ars = chi_ars[1:,:]
    np.savez('dynamic_material_simulation.npz', t=t, inc_ars=inc_ars, trans_ars=trans_ars, refl_ars=refl_ars, chi_ars=chi_ars)

# -----------------------
# Export tables and plots
# -----------------------
    
def number_formatter(num):
    num_str = '{0:.3e}'.format(np.real(num))
    e_ind = num_str.rfind('e')
    num_pre = num_str[:e_ind]
    num_pm = num_str[e_ind+1:e_ind+2]
    num_exp = num_str[e_ind+2:].strip('0')
    if num_pm == '-':
        num_exp = '-' + num_exp
    return '$' + num_pre + '\\times10^{' + num_exp + '}$' + 'SET UNITS!'

print('drude_metal simulation values')
latex_table_vals = ''
latex_table_vals += '$\\epsilon_\\infty$ & %s \\\\\n' % number_formatter(1)
latex_table_vals += '$A_{e,1}, -A_{e,2}$ & %s \\\\\n' % number_formatter(a_e)
latex_table_vals += '$A_{g,1}, -A_{g,2}$ & %s \\\\\n' % number_formatter(a_g)
latex_table_vals += '$\\gamma_e$ & %s \\\\\n' % number_formatter(gamma_e/(2*np.pi))
latex_table_vals += '$\\gamma_g$ & %s \\\\\n' % number_formatter(gamma_g/(2*np.pi))
latex_table_vals += '$\\Gamma$ & %s \\\\\n' % number_formatter(vis_fwhm)
latex_table_vals += '$\\tau_0$ & %s \\\\\n' % number_formatter(tau_decay)
latex_table_vals += '$\\tau_e$ & %s \\\\\n' % number_formatter(np.pi/gamma_e)
latex_table_vals += '$\\tau_g$ & %s \\\\\n' % number_formatter(np.pi/gamma_g)
print(latex_table_vals)
    
# Clear figure
plt.clf()
# Setup grid
cfig = plt.figure(figsize=(6.5, 4))
widths = [20, 1]
heights = [2, 1]
spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
# Add axes
axs = cfig.add_subplot(spec[1,0])
axc = cfig.add_subplot(spec[0,0], sharex=axs)
axcc = cfig.add_subplot(spec[0,1])
# Formatting
axc.set_ylabel('$t$ [ps]', fontsize=12)
axs.set_ylabel('$E_t(t)$', fontsize=12)
axs.set_xlabel('$\Delta t$ [fs]', fontsize=12)
axc.tick_params(labelsize=10, bottom=False, labelbottom=False)
axs.tick_params(labelsize=10)
axcc.tick_params(labelsize=10)
axs.set_xlim(-390, 390)
axc.set_ylim(-0.2, 0.6)
# Define variables to plot
trans_ars_to_plot = np.real(trans_ars.T)[:20000]
t_to_plot = t[:20000]
ddn = np.diff(t_diffs)[0]
time_grid, dtime_grid = np.mgrid[slice(t_to_plot[0], t_to_plot[-1] + dn, dn),
                slice(t_diffs[0],t_diffs[-1] + ddn, ddn)]
# Define colorbar
min_max = np.max(np.abs([trans_ars_to_plot.min(), trans_ars_to_plot.max()]))
cmap = plt.get_cmap('bwr')
levels = MaxNLocator(nbins=500).tick_values(-min_max, min_max)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# Plot colormesh and colorbar
im = axc.pcolormesh(dtime_grid*1e15, time_grid*1e12, trans_ars_to_plot, cmap=cmap, norm=norm)
cb = plt.colorbar(im, cax=axcc)
cb.set_label('$E_t$', fontsize=12)

# Plot lineout
n200 = np.argmin(np.abs(np.subtract(t_to_plot, -200e-15)))
n100 = np.argmin(np.abs(np.subtract(t_to_plot, -100e-15)))
zero = np.argmin(np.abs(np.subtract(t_to_plot, 0)))
p100 = np.argmin(np.abs(np.subtract(t_to_plot, 100e-15)))
p200 = np.argmin(np.abs(np.subtract(t_to_plot, 200e-15)))
axs.plot(t_diffs*1e15, np.real(trans_ars.T)[n200], label='$\Delta t=-200$fs')
axs.plot(t_diffs*1e15, np.real(trans_ars.T)[n100], label='$\Delta t=-100$fs')
axs.plot(t_diffs*1e15, np.real(trans_ars.T)[zero], label='$\Delta t=0$')
axs.plot(t_diffs*1e15, np.real(trans_ars.T)[p100], label='$\Delta t=100$fs')
axs.plot(t_diffs*1e15, np.real(trans_ars.T)[p200], label='$\Delta t=200$fs')
axs.legend(bbox_to_anchor=(1.01, 0.95), loc=2, borderaxespad=0., fontsize=8)

# Final plotting things
plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)

# Show
plt.show()