"""
Used to visualize results from the sim module
"""
from sim import Field
import numpy as np
from matplotlib import pyplot as plt


from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

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