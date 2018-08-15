
# coding: utf-8

# We start by importing the needed libraries and defining our simulation bounds and constants. Our simulation will begin at time index $-0.5$ps and end at time index $2.5$ps. The simulations spatial bounds will span from $-1$um to $2$um.

# In[1]:


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


# Specify the location of our current pulse in time and space

# In[2]:


cp_loc_val = -0.5e-6 # -0.5 um
cp_time_val = 0 # 0 fs


# Determine the simulation indicies that correspond to these locations

# In[3]:


# Find indicies
cp_loc_ind = np.argmin(np.abs(np.subtract(z, cp_loc_val)))
cp_time_ind = np.argmin(np.abs(np.subtract(t, cp_time_val)))
# Find start and end indicies in time
spread = 3500
cp_time_s = cp_time_ind - spread
cp_time_e = cp_time_ind + spread


# Create the current pulse

# In[4]:


# Make pulse
cpulse = np.append(np.diff(np.diff(np.exp(-((t[cp_time_s:cp_time_e]-cp_time_val)**2)/(8e-27)))), [0,0])
# Create Current object
current = Current(nlen, ilen, cp_time_s, cp_loc_ind, cpulse)


# Specify the location of our material (which will be $50$nm in length)

# In[5]:


# Set material length
m_len = 0.05e-6 # 50 nm
# Set locations
m_s_val = 0
m_e_val = m_s_val + m_len


# Calculate the starting and ending indicies of our material

# In[6]:


m_s_ind = np.argmin(np.abs(np.subtract(z, m_s_val)))
m_e_ind = np.argmin(np.abs(np.subtract(z, m_e_val)))
# Determine matrix length in indicies
mlen = m_e_ind - m_s_ind


# Simulate the current pulse, observing the field at the material starting index $m_0$

# In[7]:


# Run simulation if simulation save doesn't exist
sim_file = 'pre_simulation_analysis.npz'
if Path(sim_file).is_file():
    # Load results
    dat = np.load(sim_file)
    els = dat['els']
else:
    # Create Sim object, observe the field at the material start index
    s = Sim(i0, i1, di, n0, n1, dn, epsilon0, mu0, 'absorbing', current, nstore=int(nlen/50), storelocs=[m_s_ind])
    # Run simulation
    s.simulate(tqdmarg={'desc': 'Executing pre-simulation simulation', 'leave': True})
    n, ls, els, erls, hls, hrls = s.export_locs()
    np.savez(sim_file, els=els)


# Determine at what time index the pulse passes the material starting index by finding the index of the maximum value in the real values of the E-field, we will use this time to transition from non-metal to metal. Noting that the current pulse is generated at $t=0$fs and $z=-0.5$um, we would expect the current pulse to reach spatial index $z=0$um at time $t=1.67$fs as
# $$
# \frac{0.5\text{um}}{300\text{um/ps}}=1.67\text{fs}
# $$

# In[8]:


mat_transition_ind = np.argmax(np.real(els))
# Plot
plt.plot(t * 1e12, np.real(els))
plt.axvline(t[mat_transition_ind], color='k', linestyle='--')
plt.xlabel('time [ps]')
plt.ylabel('$E(t,z=m_0)$')
plt.show()
# Print time value
print('t[mat_transition_ind]=%E' % t[mat_transition_ind])


# As time steps $0.1$fs, the maximum index determined by the simulation $t=1.55$fs is one step in time away from where we would expect it to be. This is probably good enough for our purposes. Construct a spread of material transition time indicies about which to center our error function transition

# In[9]:


# Set the spread and step size
spread = int(100 / 0.1) # 100 fs / (0.1 fs/step) = 1000 steps
n_steps = 100 # Number of steps
# Calculate the step size
step_size = int(spread*2 / n_steps)
# Generate an array of the index offsets from mat_transition_ind to use
trans_inds = np.arange(-spread, spread + step_size, step_size)


# We define a function that returns an error function of the given width (in steps) and located at an offset (in steps) from `0`

# In[10]:


def normalized_error_func_gen(width, offset):
    # Normalize error function
    errfunc = (erf(np.linspace(-3,3, width)) + 1)/2
    # Determine average padding on each side
    avg_pad_amt = int(nlen/2) - int(width/2)
    l_pad = mat_transition_ind - int(width/2)
    r_pad = nlen - mat_transition_ind - int(width/2)
    # Pad error function to length of nlen
    errfunc_padded = np.pad(errfunc, (l_pad + offset, r_pad - offset), 'constant', constant_values=(0,1))
    # Return
    return errfunc_padded


# We test out this function by ofsetting the error function by plotting a $30$fs width transition offset by $10$fs.

# In[11]:


# Define width and offset
width = int(100/0.1) # 100 fs / (0.1 fs/step) = 1000 steps
offset = int(10/0.1) # 10 fs/ (0.1 fs/step) = 100 steps
# Plot
plt.plot(t * 1e15, normalized_error_func_gen(width, offset))
plt.axvline(10 + t[mat_transition_ind] * 1e15, color='k', linestyle='--')
plt.xlim((-100, 110))
plt.xlabel('time [fs]')
plt.show()


# In[12]:


def non_metal_gen(trans_ind):
    # Set constants
    a1 = np.complex64(0)
    a2 = np.complex64(1e16)
    gamma = np.complex64(1e12)
    freq = np.complex64(0)
    inf_perm = np.complex64(1e0)
    # Calculate beta
    ang_gamma = np.complex64(gamma * 2 * np.pi)
    omega = np.complex64(freq * 2 * np.pi)
    beta = np.sqrt(np.add(np.square(ang_gamma), -np.square(omega)), dtype=np.complex64)
    # Create matrices
    m = np.ones((1, mlen), dtype=np.complex64)
    mgamma = m * ang_gamma
    mbeta = m * beta
    ma1 = m * a1
    ma2 = m * a2
    # Create opacity vector using a transition width of 200 fs an offset of mat_trans_ind
    width = int(200/0.1) # 200 fs / (0.1 fs/step) = 2000 steps
    opacity = 1 - normalized_error_func_gen(width, trans_ind)
    # Create non-metal object
    return Mat(dn, ilen, nlen, m_s_ind, inf_perm, ma1, ma2, mgamma, mbeta, opacity=opacity, storelocs=[1])


# We next create our metal

# In[13]:


def metal_gen(trans_ind):
    # Set constants
    a1 = np.complex64(1e16)
    gamma = np.complex64(1e12)
    freq = np.complex64(0)
    inf_perm = np.complex64(1e0)
    # Calculate beta
    ang_gamma = np.complex64(gamma * 2 * np.pi)
    omega = np.complex64(freq * 2 * np.pi)
    beta = np.sqrt(np.add(np.square(ang_gamma), -np.square(omega)), dtype=np.complex64)
    a2 = -a1
    # Create matrices
    m = np.ones((1, mlen), dtype=np.complex64)
    mgamma = m * ang_gamma
    mbeta = m * beta
    ma1 = m * a1
    ma2 = m * a2
    # Create opacity vector using a transition width of 200 fs an offset of mat_trans_ind
    width = int(200/0.1) # 200 fs / (0.1 fs/step) = 2000 steps
    opacity = normalized_error_func_gen(width, trans_ind)
    # Create metal object
    return Mat(dn, ilen, nlen, m_s_ind, inf_perm, ma1, ma2, mgamma, mbeta, opacity=opacity, storelocs=[1])


# Create and run our simulation (or load simulation if one already exists)

# In[ ]:


#Create Sim object
sim_name = 'post_material_transition.npz'
if Path(sim_name).is_file():
    # Load results
    dat = np.load(sim_name)
    n = dat['n']
    trans_vals = dat['trans_vals']
    inc_ars = dat['inc_ars']
    trans_ars = dat['trans_ars']
    refl_ars = dat['refl_ars']
    metal_chi_ars = dat['metal_chi_ars']
    non_metal_chi_ars = dat['non_metal_chi_ars']
else:
    trans_vals = np.array([])
    # Create arrays to hold simulation values, each new simulation will contribute to a row. The zero-valued row initialized here is removed later.
    inc_ars = np.zeros((1, nlen))
    trans_ars = np.zeros((1, nlen))
    refl_ars = np.zeros((1, nlen))
    metal_chi_ars = np.zeros((1, nlen))
    non_metal_chi_ars = np.zeros((1, nlen))
    # Loop through each transition index, simulating at each
    for i in range(len(trans_inds)):
        # Get the current transition index
        trans_ind = trans_inds[i]
        # Generate materials
        non_metal = non_metal_gen(trans_ind)
        metal = metal_gen(trans_ind)
        # Create and run simulation
        s = Sim(i0, i1, di, n0, n1, dn, epsilon0, mu0, 'absorbing', current, [non_metal, metal], nstore=int(nlen/50), storelocs=[5,ilen-6])
        tqdmarg = {'desc': ('Working on transition ' + str(i+1) + '/' + str(len(trans_inds))), 'leave': False}
        s.simulate(tqdmarg)
        # Extract incident, transmitted, and reflected fields
        n, ls, els, erls, hls, hrls = s.export_locs()
        inc = erls[:,1]
        trans = els[:,1]
        refl = els[:,0] - erls[:,0]
        # Extract chi values
        ls_mat, non_metal_chi = non_metal.export_locs()
        ls_mat, metal_chi = metal.export_locs()
        # Reshape chi values
        non_metal_chi = np.reshape(non_metal_chi, (1, nlen))
        metal_chi = np.reshape(metal_chi, (1, nlen))
        # Save transition time value
        trans_vals = np.append(trans_vals, t[mat_transition_ind+trans_ind])
        # Put results into array
        inc_ars = np.vstack((inc_ars, inc))
        trans_ars = np.vstack((trans_ars, trans))
        refl_ars = np.vstack((refl_ars, refl))
        metal_chi_ars = np.vstack((metal_chi_ars, metal_chi))
        non_metal_chi_ars = np.vstack((non_metal_chi_ars, non_metal_chi))
    # Reinc_ars = move the first row of each array, which is the zero-valued row initialized earlier
    inc_ars = inc_ars[1:,:]
    trans_ars = trans_ars[1:,:]
    refl_ars = refl_ars[1:,:]
    metal_chi_ars = metal_chi_ars[1:,:]
    non_metal_chi_ars = non_metal_chi_ars[1:,:]
    # Save data
    np.savez(sim_name, n=n, trans_vals=trans_vals, inc_ars=inc_ars, trans_ars=trans_ars, refl_ars=refl_ars, metal_chi_ars=metal_chi_ars, non_metal_chi_ars=non_metal_chi_ars)


# Plot a heatmap of the field in time versus the $\Delta t$ value, which is the displacement in time from the non-metal to metal tranistion for each simulation.

# In[ ]:


# Extract values to plot
trans_ars_to_plot = np.real(trans_ars.T)

cmap = plt.get_cmap('PiYG')
levels = MaxNLocator(nbins=500).tick_values(trans_ars_to_plot.min(), trans_ars_to_plot.max())
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

ddn = np.diff(trans_vals)[0]
time_grid, dtime_grid = np.mgrid[slice(n[0], n[-1] + dn, dn),
                slice(trans_vals[0],trans_vals[-1] + ddn, ddn)]
plt.pcolormesh(dtime_grid*1e15, time_grid*1e12, trans_ars_to_plot, cmap=cmap, norm=norm)

plt.colorbar()
plt.ylabel('$t$ [ps]', fontsize=15)
plt.xlabel('$\Delta t$ [fs]', fontsize=15)
plt.title('Material Bleed Method', fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.gcf().set_dpi(400)
plt.tight_layout()
plt.savefig(fname='mat_bleed.png', format='png', dpi=600)
plt.show()

