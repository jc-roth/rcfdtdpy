"""
Used to visualize results from the sim module
"""
from sim import Field
import numpy as np
from matplotlib import pyplot as plt


from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

def contor_plot(arr, delta_z, delta_t):
    # make these smaller to increase the resolution
    # delta_z, delta_t

    # generate 2 2d grids for the x & y bounds
    t, z = np.mgrid[slice(0, 50, delta_t),
                    slice(0, 50, delta_z)]

    i = np.real(arr)

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    i = i[:-1, :-1]
    levels = MaxNLocator(nbins=5).tick_values(i.min(), i.max())


    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im = plt.pcolormesh(z, t, i, cmap=cmap, norm=norm)
    plt.colorbar(im)
    plt.title('pcolormesh with levels')
    plt.xlabel('space [i]')
    plt.ylabel('time [n]')

    plt.show()