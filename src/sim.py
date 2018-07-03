import numpy as np
from tqdm import tqdm

"""
Contains the classes used to represent a simulation
"""

class Sim:
    """
    Represents a single simulation. Field is initialized to all zeros.

    :param vacuum_permittivity: :math:`\epsilon_0`
    :param infinity_permittivity: :math:`\epsilon_\infty`
    :param vacuum_permeability: :math:`\mu_0`
    :param delta_t: :math:`\Delta t`
    :param delta_z: :math:`\Delta z`
    :param num_n: The number of time indexes
    :param num_i: The number of spatial indexes
    :param current: A current object
    :param susceptibility: A susceptibility object
    :param initial_susceptibility: The initial susceptability. Eventually will be included in the susceptibility object.
    """
    
    def __init__(self, vacuum_permittivity, infinity_permittivity, vacuum_permeability, delta_t, delta_z, num_n, num_i, current, susceptibility, initial_susceptibility):
        self._vacuum_permittivity = vacuum_permittivity
        self._infinity_permittivity = infinity_permittivity
        self._vacuum_permeability = vacuum_permeability
        self._delta_t = delta_t
        self._delta_z = delta_z
        self._current = current
        self._susceptibility = susceptibility
        self._initial_susceptibility = initial_susceptibility
        self._num_n = num_n
        self._num_i = num_i
        self._efield = Field(num_n, num_i)
        self._hfield = Field(num_n, num_i)

    def get_vacuum_permittivity(self):
        """
        Gets :math:`\epsilon_0`
        
        :returns: :math:`\epsilon_0`
        """
        return self._vacuum_permittivity

    def get_infinity_permittivity(self):
        """
        Gets :math:`\epsilon_\infty`
        
        :returns: :math:`\epsilon_\infty`
        """
        return self._infinity_permittivity

    def get_vacuum_permeability(self):
        """
        Gets :math:`\mu_0`
        
        :returns: :math:`\mu_0`
        """
        return self._vacuum_permeability

    def get_delta_t(self):
        """
        Gets :math:`\Delta t`
        
        :returns: :math:`\Delta t`
        """
        return self._delta_t

    def get_delta_z(self):
        """
        Gets :math:`\Delta z`
        
        :returns: :math:`\Delta z`
        """
        return self._delta_z

    def get_num_n(self):
        """
        Gets the number of temporal indicies.
        
        :returns: The number of temporal indicies
        """
        return self._num_n

    def get_num_i(self):
        """
        Gets the number of spatial indicies.
        
        :returns: The number of spatial indicies
        """
        return self._num_i

    def get_efield(self):
        """
        Returns the E-field as a Field object.

        :return: The E-field as a Field object
        """
        return self._efield

    def get_hfield(self):
        """
        Returns the H-field as a Field object.

        :return: The H-field as a Field object
        """
        return self._hfield
        
    def simulate(self):
        """
        Executes the simulation
        """
        # Simulate for one less step than the number of temporal indicies because initializing the fields to zero takes up the first temporal index
        for j in tqdm(range(self._num_n-1)):
            self.iterate_hfield()
            self.iterate_efield()

    def iterate_efield(self):
        r"""
        Iterates the electric field according to :math:`E^{i,n+1}=\frac{\epsilon_\infty}{\epsilon_\infty+\chi_e^0}E^{i,n}+\frac{1}{\epsilon_\infty+\chi_e^0}\psi^n+\frac{1}{\epsilon_0\left[\epsilon_\infty+\chi_e^0\right]}\frac{\Delta t}{\Delta z}\left[H^{i+1/2,n+1/2}-H^{i-1/2,n+1/2}\right]-\frac{\Delta tI_f}{\epsilon_0\left[\epsilon_\infty+\chi_e^0\right]}`. Note that the prior electric field array is located half a time index away at :math:`n-1` and the prior magnetic field array is located half a time index away at :math:`n-1/2`.
        """
        # Create an array to hold the next field iteration
        nefield = np.zeros(self._num_i, dtype=np.complex64)
        # Compute the values along the next field
        for i in range(self._num_i):
            term1 = (self._infinity_permittivity*self._efield[i])/(self._infinity_permittivity+self._initial_susceptibility)
            term2 = self.psi()/(self._infinity_permittivity+self._initial_susceptibility)
            term3 = (self._delta_t*(self._hfield[i-1]-self._hfield[i]))/(self._vacuum_permittivity*self._delta_z*(self._infinity_permittivity+self._initial_susceptibility))
            term4 = (self.current()*self._delta_t)/self._vacuum_permittivity
            nefield[i] = term1 + term2 + term3 - term4
        self._efield.update_field(nefield)
        
    def iterate_hfield(self):
        r"""
        Iterates the magnetic field according to :math:`H^{i+1/2,n+1/2}=H^{i+1/2,n-1/2}-\frac{1}{\mu_0}\frac{\Delta t}{\Delta z}\left[E^{i+1,n}-E^{i,n}\right]`. Note that the prior electric field array is located half a time index away at :math:`n-1/2` and the prior magnetic field array is located a whole time index away at :math:`n-1`.
        """
        # Create an array to hold the next field iteration
        nhfield = np.zeros(self._num_i, dtype=np.complex64)
        # Compute the values along the next field
        for i in range(self._num_i):
            term1 = self._hfield[i]
            term2 = (self._delta_t*(self._efield[i+1]-self._efield[i]))/(self._vacuum_permeability*self._delta_z)
            nhfield[i] = term1 - term2
        self._hfield.update_field(nhfield)

    def psi(self):
        """
        Calculates psi according to :math:`\psi^n=\sum^{n-1}_{m=0}E^{i,n-m}\Delta\chi_e^m` at the current time :math:`n` and position :math:`i`. Currently not implemented, and will simply return zero.

        :return: Zero
        """
        return 0

    def current(self):
        """
        Calculates the current at time :math:`n` using the current object. Currently not implemented, and will simply return zero.

        :return: Zero
        """
        return 0


class Current:
    """
    Represents a current
    """

    def __init__(self):
        pass

class Field:
    """
    Represents either an electric or magnetic field using a 2D Numpy array. The zeroth axis represents increments in time and the first axis represents increments in space.

    :param num_n: The number of temporal indexes in the field
    :param num_i: The number of spatial indexes in the field
    """

    def __init__(self, num_n, num_i):
        # Set the field time index to zero
        self._n = 0
        # Set the field temporal length to num_n
        self._num_n = num_n
        # Set the field spatial length to num_i
        self._num_i = num_i
        # Initialize a zero field
        self._field = np.zeros((num_n, num_i), dtype=np.complex64)

    def get_time_index(self):
        """
        Gets the current time index :math:`n`

        :return: The current time index :math:`n`
        """
        return self._n

    def set_time_index(self, n):
        """
        Sets the current time index to :math:`n`

        :param n: :math:`n`
        """
        # Check for that n is within the accepted range
        if(n < 0 or n >= self._num_n):
            raise IndexError('The n argument is of out of bounds')
        self._n = n

    def get_field(self, n=-1):
        """
        Gets the field at time :math:`n`, and the current time if :math:`n` is unspecified.

        :param n: :math:`n`
        :return: The field at time `n`
        """
        # If n is -1, return the current field
        if(n == -1):
            return self._field[self._n]
        # Check for that n is within the accepted range
        if(n < 0 or n >= self._num_n):
            raise IndexError('The n argument is of out of bounds')
        return self._field[n]

    def __getitem__(self, key):
        """
        Allows the [] operator to be used to get field values
        """
        return self.get_index(key)

    def get_index(self, i):
        """
        Gets the value of the field at the current time index :math:`n` and at the :math:`i` th spatial index. If the requested index is out of the field bounds, the returned value is zero.
        
        :param i: The spatial index of the field to access
        :return: The value of the field a the current time index :math:`n` and spatial index :math:`i`
        """
        # Check to see if the requested index is out of bounds, if so return zero
        if(i < 0 or i >= self._num_i):
            return np.complex64(0)
        # Return the requested field
        return self._field[self._n,i]

    def __setitem__(self, key, value):
        """
        Allows the [] operator to be used to set field values
        """
        return self.set_index(key, value)

    def set_index(self, i, value):
        """
        Sets the value of the field at the current time index :math:`n` and at the :math:`i` th spatial index.
        
        :param i: The spatial index of the field to set
        :param value: The value to set at time index :math:`n` and spatial index :math:`i`
        """
        self._field[self._n,i] = np.complex64(value)

    def update_field(self, nfield):
        """
        Iterates to the next temporal index and updates the field at that time to nfield. Raises a ValueError if the new field is not of the correct spatial length and a RuntimeError if the end of the temporal indicies has been reached.

        :param nfield: The new field to append of length num_i
        """
        # Check for nfield length, raise error if necessary
        if(len(nfield) != self._num_i):
            raise ValueError('The nfield argument is of the incorrect length, found ' + str(len(nfield)) + ', expected ' + str(self._num_i))
        # Iterate the temporal index
        self._n += 1
        # Check that the field isn't at its last time index, raise error if necessary
        if(self._n >= self._num_n):
            raise RuntimeError('End of the temporal indicies has been reached, cannot update')
        # Set the new field value
        self._field[self._n] = nfield

    def export(self):
        """
        Returns the Numpy array that contains the temporal (axis=0) and spatial values (axis=1) of the field.

        :return: A Numpy array
        """
        return self._field