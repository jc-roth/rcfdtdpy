import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    hfield = np.zeros(102)
    efield = np.zeros(102)

    xaxis = np.linspace(0, 102, 102)

    efield[51] = 1

    plt.plot(xaxis, efield, label='E')
    plt.plot(xaxis, hfield, label='H')
    plt.title(str(0))
    plt.legend()
    plt.savefig('../images/img' + str(0))
    plt.gcf().clear()

    for n in range(50):
        new_hfield = np.zeros(102)
        new_efield = np.zeros(102)
        for i in range(1, hfield.size - 1):
            new_hfield[i] = hfield[i] - (efield[i+1] - efield[i])
            new_efield[i] = efield[i] - (hfield[i] - hfield[i-1])

            if(i == 51):
                print(new_efield[i])
        hfield = new_hfield
        efield = new_efield

        plt.plot(xaxis, efield, label='E')
        plt.plot(xaxis, hfield, label='H')
        plt.title(str(n+1))
        plt.legend()
        plt.ylim((-150,150))
        plt.savefig('../images/img' + str(n+1))
        plt.gcf().clear()
