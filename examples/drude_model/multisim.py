"""
Simulates optical properties of a 10nm thick material using the Drude model in the THz.
"""

# Imports
from rcfdtd_sim import Sim, Current, Mat, vis
import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path

# Determine folder save name
fprefix = 'multisim/'
# Determine simulation config file
fsim = 'multisim.sim'


# ============================
# Material Generation Function
# ============================
def mat_gen(dn, ilen, nlen, z, a, gamma, inf_perm):
    # Set material length
    m_len = 0.010  # 10 nm

    # Set locations
    m_s_val = 0
    m_e_val = m_s_val + m_len

    # Calc start and end indices
    m_s_ind = np.argmin(np.abs(np.subtract(z, m_s_val)))
    m_e_ind = np.argmin(np.abs(np.subtract(z, m_e_val)))
    # Set constants
    a = np.complex64(a)
    gamma = np.complex64(gamma)
    nu = np.complex64(0)

    # Calculate more constants
    ang_gamma = np.complex64(gamma * 2 * np.pi)
    omega = np.complex64(nu * 2 * np.pi)
    beta = np.sqrt(np.add(np.square(ang_gamma), -np.square(omega)), dtype=np.complex64)
    a1 = np.complex64(a)
    a2 = np.complex64(-a)

    # Construct matrices
    mlen = m_e_ind - m_s_ind
    m = np.ones((1, mlen), dtype=np.complex64)
    mgamma = m * ang_gamma
    mbeta = m * beta
    ma1 = m * a1
    ma2 = m * a2
    material = Mat(dn, ilen, nlen, m_s_ind, inf_perm, ma1, ma2, mgamma, mbeta, storelocs=[1])
    return material


# =============
# Define main()
# =============
def main():

    # NOTIFY USER
    print('Preparing simulation')

    # PREPARE SIMULATION
    # Constants
    c0 = 1  # um/ps
    di = 0.003  # 0.003 um
    dn = di/c0  # (0.003 um) / (300 um/ps) = 0.00001 ps = 0.01 fs
    epsilon0 = 1
    mu0 = 1

    # Define bounds
    i0 = -1  # -1 um
    i1 = 2  # 2 um
    n0 = -225  # (0.01 fs) * (-225 um) / (0.003 um/step) = (0.01 fs) * (-75,000 steps) = -750 fs = -0.75 ps
    n1 = 300  # (0.01 fs) * (300 um) / (0.003 um/step) = (0.01 fs) * (100,000 steps) = 1,000 fs = 1 ps

    # Calculate dimensions
    nlen, ilen = Sim.calc_dims(n0, n1, dn, i0, i1, di)
    # Create a arrays that hold the value of the center of each cell
    t = np.linspace(n0+dn/2, n1+dn/2, nlen, endpoint=False) * (10/3)  # Multiply by 10/3 to get from um -> fs
    z = np.linspace(i0+di/2, i1+di/2, ilen, endpoint=False)

    # PREPARE CURRENT
    # Define locations
    cp_loc_val = -0.5 # -0.5 um
    cp_time_val = 0 # 0 fs

    # Find indicies
    cp_loc_ind = np.argmin(np.abs(np.subtract(z, cp_loc_val)))
    cp_time_ind = np.argmin(np.abs(np.subtract(t, cp_time_val)))

    # Find start and end indicies in time
    spread = int(500 / 0.01) # (500 fs) / (0.1 fs/step) = 5,000 steps
    cp_time_s = cp_time_ind - spread
    cp_time_e = cp_time_ind + spread

    # Make pulse
    cpulse = np.append(np.diff(np.diff(np.exp(-((t[cp_time_s:cp_time_e]-cp_time_val)**2)/(2e4)))), [0,0])

    # Create Current object
    current = Current(nlen, ilen, cp_time_s, cp_loc_ind, cpulse)

    # LOAD REQUESTED SIMULATIONS
    sims = np.loadtxt(fsim, dtype=str, delimiter=',', skiprows=1)
    for a, gamma, inf_perm, fname in sims:
        # Parse floats
        a = np.float(a)
        gamma = np.float(gamma)
        inf_perm = np.float(inf_perm)
        # Check to see if simulation already exists
        simfile = Path(fprefix + fname + '.npz')
        # Notify user
        print('Checking simulation with parameters a=%10.10f gamma=%10.10f infperm=%10.10f' % (a, gamma, inf_perm))
        if simfile.is_file():
            print('Simulation exists, loading now...')
            # Load results
            dat = np.load(simfile)
            n = dat['n']
            ls = dat['ls']
            els = dat['els']
            erls = dat['erls']
            hls = dat['hls']
            hrls = dat['hrls']
            chi = dat['chi']
        else:
            tqdmarg = {'leave': False, 'desc': 'Simulation does not exist, simulating now...'}
            # Finish preparing simulation
            mat = mat_gen(dn, ilen, nlen, z, a, gamma, inf_perm)
            s = Sim(i0, i1, di, n0, n1, dn, epsilon0, mu0, 'absorbing', current, mat, storelocs=[5,ilen-6])
            # Run simulation
            s.simulate(tqdmarg)
            # Notify user
            print('Done simulating')
            # Export and save arrays
            n, ls, els, erls, hls, hrls = s.export_locs()
            ls_mat, chi = mat.export_locs()
            n = n * (10/3) # 10/3 scale factor converts from um -> fs
            np.savez(fprefix+fname+'.npz', n=n, ls=ls, els=els, erls=erls, hls=hls, hrls=hrls, chi=chi)

        # CALCULATIONS
        # Extract incident, transmitted, and reflected fields
        inc = erls[:,1]
        trans = els[:,1]
        refl = els[:,0] - erls[:,0]

        # Calculate time difference
        dt = np.diff(n)[0] # Calculate time step difference in fs

        # Calculate Fourier transforms
        freq = fftfreq(nlen, dt) * 1e3 # in THz (since [dt]=[fs], 1/[dt] = 1/[fs] = 10^15/[s] = 10^3*10^12/[s] = 10^4*[THz])
        incf = fft(inc)
        transf = fft(trans)

        # Removeunwanted frequencies
        freq = freq[1:int(nlen/2)]
        incf = incf[1:int(nlen/2)]
        transf = transf[1:int(nlen/2)]

        # Remove incf zero indicies from all arrays
        nonzero_ind = np.nonzero(incf)
        freq = freq[nonzero_ind]
        incf = incf[nonzero_ind]
        transf = transf[nonzero_ind]

        # Calculate spectrum in frequency
        spec = np.divide(transf, incf)

        # Remove spec zero indicies from all arrays
        nonzero_ind = np.nonzero(spec)
        freq = freq[nonzero_ind]
        incf = incf[nonzero_ind]
        transf = transf[nonzero_ind]
        spec = spec[nonzero_ind]

        # Extract phase and magnitude
        spec_m = np.absolute(spec)
        spec_a = np.abs(np.unwrap(np.angle(spec)))

        # Set calculation constants
        L = 10 * 1e-9
        Z0 = 376.73 # Ohms (impedance of free space)
        permittivity_free_space = 8.854187817e-12

        # Calculate the angular frequency
        ang_freq = 2 * np.pi * freq # THz * 2pi

        # Calculate conductivity
        conductivity = np.multiply(np.divide(2, Z0*L), np.subtract(np.divide(1, spec), 1))

        # Calculate index of refraction
        n_complex = np.sqrt(inf_perm + np.divide(np.multiply(1j, conductivity), np.multiply(ang_freq, permittivity_free_space)))

        # Calculate the imaginary part of the index of refraction
        n_real = np.real(n_complex)
        n_imag = np.imag(n_complex)

        # PLOTTING
        spec_real = np.real(spec)
        spec_imag = np.imag(spec)

        f_max = np.argmin(np.abs(np.subtract(freq, 8)))

        r_max = np.max(spec_real[0:f_max])
        r_min = np.min(spec_real[0:f_max])

        i_max = np.max(spec_imag[0:f_max])
        i_min = np.min(spec_imag[0:f_max])

        plt.close('all')
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
        ax0.set_title(r'RC-FDTD Simulation: $\epsilon_\infty=%5.5f$, $A=%5.5f$, $\gamma=%5.5f$, $L=10$nm, data=%s.npz' % (inf_perm, a, gamma, fname))
        ax1.set_xlim(0, 8)
        ax0.plot(freq, spec_real)
        ax0.set_ylim(r_min, r_max)
        ax0.set_ylabel('Real')
        ax1.plot(freq, spec_imag)
        ax1.set_ylim(i_min, i_max)
        ax1.set_ylabel('Imaginary')
        ax1.set_xlabel('Frequency')
        fig.savefig(fname=fprefix+fname+'.pdf', format='pdf')

        # PLOTTING
        """
        # Setup figure
        plt.close('all')
        fig = plt.figure(figsize=(12, 8), dpi=60)

        # Setup axes
        ax_chi = plt.subplot2grid((5,3), (1, 0), 2, 1)
        ax_freq = plt.subplot2grid((5,3), (1, 1), 2, 2)
        ax_time = plt.subplot2grid((5,3), (0, 0), 1, 3)
        ax_t = plt.subplot2grid((5,3), (3, 0), 1, 2, sharex = ax_freq)
        ax_p = plt.subplot2grid((5,3), (4, 0), 1, 2, sharex = ax_t)
        ax_conductivity = plt.subplot2grid((5,3), (3, 2), 2, 1)

        # Time axis
        ax_time.plot(n*1e-3, inc, label='$E_i(t)$')
        ax_time.plot(n*1e-3, trans, label='$E_t(t)$')
        ax_time.plot(n*1e-3, refl, label='$E_r(t)$')
        ax_time.set_ylabel('amplitude [?]')
        ax_time.set_xlabel('time [ps]')
        ax_time.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)

        # Chi axis
        ax_chi.plot(n,chi)
        ax_chi.set_ylabel('current [?]')
        ax_chi.set_xlabel('time [fs]')
        ax_chi.set_xlim(-750, -700)
        ax_chi.ticklabel_format(style='sci', scilimits=(0,0), axis='y')

        # Frequency axis
        ax_freq.plot(freq, np.abs(incf), label='$E_i(\omega)$')
        ax_freq.plot(freq, np.abs(transf), label='$E_t(\omega)$')
        ax_freq.set_ylabel('amplitude [?]')
        ax_freq.set_xlabel('frequency [THz]')
        ax_freq.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)

        # Coefficient plot
        ax_t.plot(freq, spec_m)
        ax_t.set_ylabel(r'$E_t(\omega)/E_i(\omega)$')
        ax_t.set_ylim(0, 2)

        ax_p.plot(freq, spec_a)
        ax_p.set_ylabel(r'$\phi(\nu)$ [rad]')
        ax_p.set_xlabel(r'frequency [THz]')
        ax_p.ticklabel_format(style='sci', scilimits=(0,0), axis='y')

        ax_p.set_xlim(0, 10)

        # Conductivity plot

        conductivity_real_line, = ax_conductivity.plot(ang_freq, np.real(conductivity), 'b-')
        ax_conductivity.set_xlabel(r'$\omega$ [$2\pi\times$THz]')
        ax_conductivity.set_ylabel(r'$\mathcal{R}(\sigma)$')
        # ax_n.legend((n_real_line, n_imag_line), ('$n$', '$\kappa$'), bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode='expand', borderaxespad=0)
        ax_conductivity.set_xlim(0, 2*np.pi*3.5)
        ax_conductivity.set_yscale('log')
        ax_conductivity.set_ylim((0,6e4))

        # Final setup
        ax_time.set_title(r'RC-FDTD Simulation: $\epsilon_\infty=%5.5f$, $A=%5.5f$, $\gamma=%5.5f$, $L=10$nm, data=%s.npz' % (inf_perm, a, gamma, fname))
        plt.tight_layout()

        # Show figure
        # fig.savefig(fname=fprefix+fname+'.pdf', format='pdf')
        plt.show()
        """


# Run if main script
if __name__ == '__main__':
    main()
