"""
Used to quickly visualize results from the sim module, should not be used for production quality plots
"""
from .sim import Simulation

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def plot(sim, fname=None, nscale=1, escale=1, hscale=1, nunit='NA', eunit='NA', hunit='NA'):
    """
    Displays the E and H-fields at a given index in time.

    :param sim: The simulation to visualize
    :param fname: The filename to export to, does not export if blank
    :param nscale: The scalar factor that the x-axis is scaled by
    :param escale: The scalar factor that the y-axis of the E-field is scaled by
    :param hscale: The scalar factor that the y-axis of the H-field is scaled by
    :param nunit: The units given to the x-label
    :param eunit: The units given to the y-label for the E-field axis
    :param hunit: The units given to the y-label for the H-field axis
    """
    # Export variables from the simulation
    n, ls, els, erls, hls, hrls = sim.export_locs()
    # Populate loc if no lines are specified
    nlocs = np.shape(els)[1]
    # Apply scale factors and take real parts
    n *= nscale
    els = np.real(els)*escale
    hls = np.real(hls)*hscale
    erls = np.real(erls)*escale
    hrls = np.real(hrls)*hscale
    # Create a new figure and set x-limits
    fig, [ax0, ax1] = plt.subplots(2, 1, True)
    ax0.set_xlim(n[0], n[-1])
    ax1.set_xlim(n[0], n[-1])
    # Prepare max values
    emax = 0
    hmax = 0
    for l in range(nlocs):
        # Extract relevant info
        e = els[:,l]
        er = erls[:,l]
        h = hls[:,l]
        hr = hrls[:,l]
        # Update the y-axis limits
        loc_emax = max(np.abs([np.max(e), np.min(e), np.max(er), np.min(er)]))
        loc_hmax = max(np.abs([np.max(h), np.min(h), np.max(hr), np.min(hr)]))
        emax = max([loc_emax, emax])
        hmax = max([loc_hmax, hmax])
        # Plot
        ax0.plot(n, e, label='$E^{'+str(ls[l])+',n}$', linestyle='-')
        ax0.plot(n, er, label='$E^{'+str(ls[l])+',n}$ reference', linestyle='--')
        # Note the np.diff() function is used to offset the H-field plot as required by the Yee cell
        ax1.plot(n+np.diff(n[0:2]), h, label='$H^{'+str(ls[l])+',n}$', linestyle='-')
        ax1.plot(n+np.diff(n[0:2]), hr, label='$H^{'+str(ls[l])+',n}$ reference', linestyle='--')
    # Set axis limits
    ax0.set_ylim(emax*1.1, -emax*1.1)
    ax1.set_ylim(hmax*1.1, -hmax*1.1)
    # Add legends
    ax0.legend(loc=1)
    ax1.legend(loc=1)
    # Label axes
    ax0.set_ylabel('$E$ [%s]' % eunit)
    ax1.set_ylabel('$H$ [%s]' % hunit)
    ax1.set_xlabel('$t$ [%s]' % nunit)
    # Final preparations
    plt.tight_layout()
    # Display or save
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

def timeseries(sim, fname=None, interval=10, iscale=1, escale=1, hscale=1, iunit='NA', eunit='NA', hunit='NA'):
    """
    Animates the E and H-fields in time.

    :param sim: The simulation to visualize
    :param interval: The interval between timesteps in milliseconds
    :param fname: The filename to export to, does not export if blank
    :param iscale: The scalar factor that the x-axis is scaled by
    :param escale: The scalar factor that the y-axis of the E-field is scaled by
    :param hscale: The scalar factor that the y-axis of the H-field is scaled by
    :param iunit: The units given to the x-label
    :param eunit: The units given to the y-label for the E-field axis
    :param hunit: The units given to the y-label for the H-field axis
    """
    # Export variables from the simulation
    n, i, e, h, er, hr = sim.export_fields()
    # Take real parts
    e = np.real(e)
    h = np.real(h)
    er = np.real(er)
    hr = np.real(hr)
    # Apply scale factors
    i *= iscale
    e *= escale
    h *= hscale
    er *= escale
    hr *= hscale
    # Create a new figure and set x-limits
    fig = plt.figure()
    ax0 = plt.axes()
    ax0.set_xlim(i[0], i[-1])
    # Determine y-axis limits
    emax = max(np.abs([np.max(e), np.min(e), np.max(er), np.min(er)])) * 1.1
    hmax = max(np.abs([np.max(h), np.min(h), np.max(hr), np.min(hr)])) * 1.1
    # Create the second axis
    ax1 = ax0.twinx()
    # Set axis limits
    ax0.set_ylim(-emax, emax)
    ax1.set_ylim(-hmax, hmax)
    # Plot
    le, = ax0.plot([], [], color='#1f77b4', linestyle='-')
    ler, = ax0.plot([], [], color='#1f77b4', linestyle='--')
    lh, = ax1.plot([], [], color='#ff7f0e', linestyle='-')
    lhr, = ax1.plot([], [], color='#ff7f0e', linestyle='--')
    # Add legend
    ax0.legend((le, lh, ler, lhr), ('E', 'H', 'E reference', 'H reference'), loc=1)
    # Label axes
    ax0.set_xlabel('$z$ [%s]' % iunit)
    ax0.set_ylabel('$E$ [%s]' % eunit)
    ax1.set_ylabel('$H$ [%s]' % hunit)
    # Final preparations
    plt.tight_layout()
    # Define the initialization and update functions
    def init():
        le.set_data([], [])
        ler.set_data([], [])
        lh.set_data([], [])
        lhr.set_data([], [])
        return (le,ler,lh,lhr)
    def update(n):
        le.set_data(i, e[n])
        ler.set_data(i, er[n])
        lh.set_data(i+np.diff(i[0:2]), h[n]) # Note the np.diff() function is used to offset the H-field plot as required by the Yee cell
        lhr.set_data(i+np.diff(i[0:2]), hr[n]) # Note the np.diff() function is used to offset the H-field plot as required by the Yee cell
        return (le,ler,lh,lhr)
    # Run animation
    anim = animation.FuncAnimation(fig, update, frames=np.shape(e)[0], interval=interval, init_func=init, blit=True)
    # Display or save
    if fname is None:
        plt.show()
    else:
        anim.save(fname, fps=int(1000/interval))