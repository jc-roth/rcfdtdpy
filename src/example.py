from sim import Sim
from sim import Field
import numpy as np
"""
A module that shows how to use the sim module to prepare and run RC-FDTD simulations.
"""

def prep_sim():
    """
    Used to prepare the simulation object
    """

if __name__ == '__main__':
    cfield = Field(50, 50)
    cfield[2] = 5
    s = Sim(1,1,1,1,1,50,50,cfield,0,0)
    s.simulate()
    arr = s.get_efield().export()
    for i in range(50):
        print(str(i) + ':\t' + str(arr[i]))