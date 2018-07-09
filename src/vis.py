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

def plot(sim, n):
    """
    Displays the E and H-fields at a given index in time.

    :param sim: The simulation to export
    :param n: The temporal index :math:`n` to display
    """
    # Export variables from the simulation
    n, i, e, h, c = sim.export()
    # Create a new figure and place lines on it
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(i, e[n], label='E')
    ax.plot(i, h[n], label='H')
    ax.legend(loc=1)
    # Determine axis limits
    ax.set_xlim(i[0], i[-1])
    emax = max(np.abs([np.max(e), np.min(e)]))
    hmax = max(np.abs([np.max(h), np.min(h)]))
    ylim = max(emax, hmax) * 1.1
    ax.set_ylim(ylim, -ylim)
    # Label axes
    ax.set_ylabel('Field Amplitude [UNITS?]')
    ax.set_xlabel('$z$ [UNITS?]')
    # Run animation
    plt.show()

def timeseries(sim, interval=10, fname=None):
    """
    Animates the E and H-fields in time.

    :param sim: The simulation to export
    :param interval: The interval between timesteps in milliseconds
    :param fname: The filename to export to, does not export if blank
    """
    # Export variables from the simulation
    n, i, e, h, c = sim.export()
    # Create a new figure and place lines on it
    fig = plt.figure()
    ax = plt.axes()
    le, = ax.plot([], [], label='E')
    lh, = ax.plot([], [], label='H')
    ax.legend(loc=1)
    # Determine axis limits
    ax.set_xlim(i[0], i[-1])
    emax = max(np.abs([np.max(e), np.min(e)]))
    hmax = max(np.abs([np.max(h), np.min(h)]))
    ylim = max(emax, hmax) * 1.1
    ax.set_ylim(ylim, -ylim)
    # Label axes
    ax.set_ylabel('Field Amplitude [UNITS?]')
    ax.set_xlabel('$z$ [UNITS?]')
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
    if fname is None:
        plt.show()
    else:
        anim.save(fname, fps=int(1000/interval))

def contor_plot(e, h):
    # make these smaller to increase the resolution
    # delta_z, delta_t
    
    # generate 2 2d grids for the x & y bounds
    dim_t, dim_z = np.shape(e)
    t, z = np.mgrid[slice(0, dim_t, 1),
                    slice(0, dim_z, 1)]

    ie = np.real(e)
    ih = np.real(h)

    fig, axes = plt.subplots(nrows=2, sharex=True)

    # t and z are bounds, so i should be the value *inside* those bounds.
    # Therefore, remove the last value from the i array.
    ie = ie[:-1, :-1]
    levelse = MaxNLocator(nbins=500).tick_values(ie.min(), ie.max())


    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmape = plt.get_cmap('PiYG')
    norme = BoundaryNorm(levelse, ncolors=cmape.N, clip=True)

    ime = axes[0].pcolormesh(z, t, ie, cmap=cmape, norm=norme)
    fig.colorbar(ime, ax=axes[0])
    axes[0].set_ylabel('E\ntime [n]')

    # t and z are bounds, so i should be the value *inside* those bounds.
    # Therefore, remove the last value from the i array.
    ih = ih[:-1, :-1]
    levelsh = MaxNLocator(nbins=500).tick_values(ih.min(), ih.max())


    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmaph = plt.get_cmap('PiYG')
    normh = BoundaryNorm(levelsh, ncolors=cmaph.N, clip=True)

    imh = axes[1].pcolormesh(z, t, ih, cmap=cmaph, norm=normh)
    fig.colorbar(imh, ax=axes[1])
    axes[1].set_xlabel('space [i]')
    axes[1].set_ylabel('H\ntime [n]')

    plt.show()