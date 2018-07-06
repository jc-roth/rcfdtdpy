import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    time_dim = 300
    space_dim = 102

    hfield = np.zeros(space_dim)
    efield = np.zeros(space_dim)

    current = np.zeros((time_dim, space_dim))

    x = np.arange(0, time_dim+2, 1)

    current[:, 49] = np.diff(np.diff(np.exp(-((x-20)**2)/(5))))

    xaxis = np.linspace(0, 102, 102)

    plt.plot(xaxis, efield, label='E')
    plt.plot(xaxis, hfield, label='H')
    plt.title(str(0))
    plt.legend()
    plt.savefig('../temp/img' + str(0))
    plt.gcf().clear()

    for n in range(time_dim):
        new_hfield = np.zeros(102)
        new_efield = np.zeros(102)
        for i in range(1, hfield.size - 1):
            hfield[i] = hfield[i] - (efield[i+1] - efield[i])
            efield[i] = efield[i] - (hfield[i] - hfield[i-1]) - current[n, i]

            if(i == 49):
                print(efield[i])
        #hfield = new_hfield
        #efield = new_efield

        plt.plot(xaxis, efield, label='E')
        plt.plot(xaxis, hfield, label='H')
        plt.title(str(n+1))
        plt.ylim((-0.3, 0.3))
        plt.legend()
        plt.savefig('../temp/img' + str(n+1))
        plt.gcf().clear()
