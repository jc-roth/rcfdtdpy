"""
Used to visualize results from the sim module
"""
from sim import Sim

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

def plot(sim, n, fname=None, iscale=1, escale=1, hscale=1, iunit='NA', eunit='NA', hunit='NA'):
    """
    Displays the E and H-fields at a given index in time.

    :param sim: The simulation to visualize
    :param n: The temporal index :math:`n` to display
    :param fname: The filename to export to, does not export if blank
    :param iscale: The scalar factor that the x-axis is scaled by
    :param escale: The scalar factor that the y-axis of the E-field is scaled by
    :param hscale: The scalar factor that the y-axis of the H-field is scaled by
    :param iunit: The units given to the x-label
    :param eunit: The units given to the y-label for the E-field axis
    :param hunit: The units given to the y-label for the H-field axis
    """
    # Export variables from the simulation
    n, i, e, h, er, hr, c = sim.export()
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
    ax0.set_ylim(emax, -emax)
    ax1.set_ylim(hmax, -hmax)
    # Plot
    le = ax0.plot(i, e[n], color='#1f77b4', linestyle='-')
    ler = ax0.plot(i, er[n], color='#1f77b4', linestyle='--')
    lh = ax1.plot(i+np.diff(i[0:2]), h[n], color='#ff7f0e', linestyle='-') # Note the np.diff() function is used to offset the H-field plot as required by the Yee cell
    lhr = ax1.plot(i+np.diff(i[0:2]), hr[n], color='#ff7f0e', linestyle='--') # Note the np.diff() function is used to offset the H-field plot as required by the Yee cell
    # Add legend
    ax0.legend((le + lh + ler + lhr), ('E', 'H', 'E reference', 'H reference'), loc=1)
    # Label axes
    ax0.set_xlabel('$z$ [%s]' % iunit)
    ax0.set_ylabel('$E$ [%s]' % eunit)
    ax1.set_ylabel('$H$ [%s]' % hunit)
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
    n, i, e, h, er, hr, c = sim.export()
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
    anim = animation.FuncAnimation(fig, update, frames=len(n), interval=interval, init_func=init, blit=True)
    # Display or save
    if fname is None:
        plt.show()
    else:
        anim.save(fname, fps=int(1000/interval))

def contor(sim, fname=None, nlevels=500, nscale=1, iscale=1, escale=1, hscale=1, nunit='NA', iunit='NA', eunit='NA', hunit='NA'):
    """
    Displays the E and H-fields on a contor plot

    :param sim: The simulation to visualize
    :param fname: The filename to export to, does not export if blank
    :param nlevels: The number of color levels to display the field intensities with
    :param iscale: The scalar factor that the x-axis is scaled by
    :param escale: The scalar factor that the y-axis of the E-field is scaled by
    :param hscale: The scalar factor that the y-axis of the H-field is scaled by
    :param iunit: The units given to the x-label
    :param eunit: The units given to the y-label for the E-field axis
    :param hunit: The units given to the y-label for the H-field axis
    """
    # Extract simulation results and parameters
    n, i, e, h, c = sim.export()
    n0, n1, dn, i0, i1, di = sim.get_bound_res()
    # Apply scale factors
    i *= iscale
    n *= nscale
    e *= escale
    h *= hscale
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
    axes[0].set_ylabel('$E$ - $t$ [%s]' % nunit)
    # Plot the H-field
    hmap = axes[1].pcolormesh(igrid, ngrid, h, cmap=cmap, norm=norm)
    fig.colorbar(hmap, ax=axes[1])
    axes[1].set_xlabel('$z$ [%s]' % iunit)
    axes[1].set_ylabel('$H$ - $t$ [%s]' % nunit)
    # Display or save
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)