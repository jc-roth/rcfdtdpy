import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    time_dim = 50
    space_dim = 102

    hfield = np.zeros(space_dim)
    efield = np.zeros(space_dim)

    current = np.zeros((time_dim, space_dim))

    for n in range(time_dim):
        current[n, 49] = np.sin(2*np.pi*(n/15))

    xaxis = np.linspace(0, 102, 102)

    plt.plot(xaxis, efield, label='E')
    plt.plot(xaxis, hfield, label='H')
    plt.title(str(0))
    plt.legend()
    plt.savefig('../images/img' + str(0))
    plt.gcf().clear()

    for n in range(time_dim):
        new_hfield = np.zeros(102)
        new_efield = np.zeros(102)
        for i in range(1, hfield.size - 1):
            new_hfield[i] = hfield[i] - (efield[i+1] - efield[i])
            new_efield[i] = efield[i] - (hfield[i] - hfield[i-1]) - current[n, i]

            if(i == 51):
                print(new_efield[i])
        hfield = new_hfield
        efield = new_efield

        plt.plot(xaxis, efield, label='E')
        plt.plot(xaxis, hfield, label='H')
        plt.plot(xaxis, 100*current[n], label=r'C [$\times 100$]')
        plt.title(str(n+1))
        plt.legend()
        plt.ylim((-150,150))
        plt.savefig('../temp/img' + str(n+1))
        plt.gcf().clear()
