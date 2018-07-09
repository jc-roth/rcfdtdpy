"""
Used to visualize results from the sim module
"""
from sim import Field
from sim import Sim

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

def plot(sim, n, fname=None, xscale=1, yscale_e=1, yscale_h=1, xunit='NA', yunit_e='NA', yunit_h='NA'):
    """
    Displays the E and H-fields at a given index in time.

    :param sim: The simulation to visualize
    :param n: The temporal index :math:`n` to display
    :param fname: The filename to export to, does not export if blank
    :param xscale: The scalar factor that the x-axis is scaled by
    :param yscale_e: The scalar factor that the y-axis of the E-field is scaled by
    :param yscale_h: The scalar factor that the y-axis of the H-field is scaled by
    :param xunit: The units given to the x-label
    :param yunit_e: The units given to the y-label for the E-field axis
    :param yunit_h: The units given to the y-label for the H-field axis
    """
    # Export variables from the simulation
    n_arr, i, e, h, c = sim.export()
    # Apply scale factors
    i *= xscale
    e *= yscale_e
    h *= yscale_h
    # Create a new figure and set x-limits
    fig = plt.figure()
    ax0 = plt.axes()
    ax0.set_xlim(i[0], i[-1])
    # Determine y-axis limits
    emax = max(np.abs([np.max(e[n]), np.min(e[n])])) * 1.1
    hmax = max(np.abs([np.max(h[n]), np.min(h[n])])) * 1.1
    # Create the second axis
    ax1 = ax0.twinx()
    # Set axis limits
    ax0.set_ylim(emax, -emax)
    ax1.set_ylim(hmax, -hmax)
    # Plot
    le = ax0.plot(i, e[n], color='#1f77b4')
    lh = ax1.plot(i, h[n], color='#ff7f0e')
    # Add legend
    ax0.legend((le + lh), ('E', 'H'), loc=1)
    # Label axes
    ax0.set_xlabel('$z$ [%s]' % xunit)
    ax0.set_ylabel('$E$ [%s]' % yunit_e)
    ax1.set_ylabel('$H$ [%s]' % yunit_h)
    # Final preparations
    plt.tight_layout()
    # Display or save
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

def timeseries(sim, fname=None, interval=10, xscale=1, yscale_e=1, yscale_h=1, xunit='NA', yunit_e='NA', yunit_h='NA'):
    """
    Animates the E and H-fields in time.

    :param sim: The simulation to visualize
    :param interval: The interval between timesteps in milliseconds
    :param fname: The filename to export to, does not export if blank
    """
    # Export variables from the simulation
    n, i, e, h, c = sim.export()
    # Apply scale factors
    i *= xscale
    e *= yscale_e
    h *= yscale_h
    # Create a new figure and set x-limits
    fig = plt.figure()
    ax0 = plt.axes()
    ax0.set_xlim(i[0], i[-1])
    # Determine y-axis limits
    emax = max(np.abs([np.max(e), np.min(e)])) * 1.1
    hmax = max(np.abs([np.max(h), np.min(h)])) * 1.1
    # Create the second axis
    ax1 = ax0.twinx()
    # Set axis limits
    ax0.set_ylim(-emax, emax)
    ax1.set_ylim(-hmax, hmax)
    # Plot
    le, = ax0.plot([], [], color='#1f77b4')
    lh, = ax1.plot([], [], color='#ff7f0e')
    # Add legend
    ax0.legend((le, lh), ('E', 'H'), loc=1)
    # Label axes
    ax0.set_xlabel('$z$ [%s]' % xunit)
    ax0.set_ylabel('$E$ [%s]' % yunit_e)
    ax1.set_ylabel('$H$ [%s]' % yunit_h)
    # Final preparations
    plt.tight_layout()
    # Define the initialization and update functions
    def init():
        le.set_data([], [])
        lh.set_data([], [])
        return (le,lh)
    def update(n):
        le.set_data(i, e[n])
        lh.set_data(i, h[n])
        return (le,lh)
    # Run animation
    anim = animation.FuncAnimation(fig, update, frames=len(n), interval=interval, init_func=init, blit=True)
    # Display or save
    if fname is None:
        plt.show()
    else:
        anim.save(fname, fps=int(1000/interval))

def contor(sim, fname=None, nlevels=500):
    """
    Displays the E and H-fields on a contor plot

    :param sim: The simulation to visualize
    :param fname: The filename to export to, does not export if blank
    :param nlevels: The number of color levels to display the field intensities with
    """
    # Extract simulation results and parameters
    n, i, e, h, c = sim.export()
    n0, n1, dn, i0, i1, di = sim.get_bound_res()
    # Generate mesh grid
    ngrid, igrid = np.mgrid[n0:n1:dn, i0:i1:di]
    # Create figure and axes
    fig, axes = plt.subplots(nrows=2, sharex=True)
    # The e-field and h-fields have to fit into the bounds, so we must shrink them by one cell on each dimension
    e = e[:-1, :-1]
    h = h[:-1, :-1]
    # Determine the levels to use
    lmin = min([np.min(e), np.min(h)])
    lmax = max([np.max(e), np.max(h)])
    levels = MaxNLocator(nbins=nlevels).tick_values(lmin, lmax)
    # Setup the colormap
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    # Plot the E-field
    emap = axes[0].pcolormesh(igrid, ngrid, e, cmap=cmap, norm=norm)
    fig.colorbar(emap, ax=axes[0])
    axes[0].set_ylabel('E\ntime [n]')
    # Plot the H-field
    hmap = axes[1].pcolormesh(igrid, ngrid, h, cmap=cmap, norm=norm)
    fig.colorbar(hmap, ax=axes[1])
    axes[1].set_xlabel('space [i]')
    axes[1].set_ylabel('H\ntime [n]')
    # Display or save
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)