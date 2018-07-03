import numpy as np

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
        self._efield = Field(num_i)
        self._hfield = Field(num_i)

    def set_vacuum_permittivity(self, vacuum_permittivity):
        """
        Sets :math:`\epsilon_0`

        :param vacuum_permittivity: :math:`\epsilon_0`
        """
        self._vacuum_permittivity = vacuum_permittivity

    def get_vacuum_permittivity(self):
        """
        Gets :math:`\epsilon_0`
        
        :returns: :math:`\epsilon_0`
        """
        return self._vacuum_permittivity

    def set_infinity_permittivity(self, infinity_permittivity):
        """
        Sets :math:`\epsilon_\infty`

        :param infinity_permittivity: :math:`\epsilon_\infty`
        """
        self._infinity_permittivity = infinity_permittivity

    def get_delta_t(self):
        """
        Gets :math:`\Delta t`
        
        :returns: :math:`\Delta t`
        """
        return self._delta_t

    def set_delta_t(self, delta_t):
        """
        Sets :math:`\Delta t`
        
        :param delta_t: :math:`\Delta t`
        """
        self._delta_t = delta_t

    def get_delta_z(self):
        """
        Gets :math:`\Delta z`
        
        :returns: :math:`\Delta z`
        """
        return self._delta_z

    def set_delta_z(self, delta_z):
        """
        Sets :math:`\Delta z`
        
        :param delta_z: :math:`\Delta z`
        """
        self._delta_z = delta_z

    def iterate_efield(self, pef, phf):
        r"""
        Iterates the electric field according to :math:`E^{i,n+1}=\frac{\epsilon_\infty}{\epsilon_\infty+\chi_e^0}E^{i,n}+\frac{1}{\epsilon_\infty+\chi_e^0}\psi^n+\frac{1}{\epsilon_0\left[\epsilon_\infty+\chi_e^0\right]}\frac{\Delta t}{\Delta z}\left[H^{i+1/2,n+1/2}-H^{i-1/2,n+1/2}\right]-\frac{\Delta tI_f}{\epsilon_0\left[\epsilon_\infty+\chi_e^0\right]}`
        
        :param pef: The prior electric field array (located half a time index away at :math:`n-1`)
        :param phf: The prior magnetic field array (located half a time index away at :math:`n-1/2`)
        """
        # Create an array to hold the next field iteration
        nefield = np.zeros(self._num_i, dtype=np.complex64)
        # Compute the values along the next field
        for i in range(self._num_i):
            term1 = (self._infinity_permittivity*pef[i])/(self._infinity_permittivity+self._initial_susceptibility)
            term2 = self.psi()/(self._infinity_permittivity+self._initial_susceptibility)
            term3 = (self._delta_t*(phf[i-1]-phf[i]))/(self._vacuum_permittivity*self._delta_z*(self._infinity_permittivity+self._initial_susceptibility))
            term4 = (self.current()*self._delta_t)/self._vacuum_permittivity
            nefield[i] = term1 + term2 + term3 - term4
        self._efield.append_field(nefield)
        
    def iterate_hfield(self, pef, phf):
        r"""
        Iterates the magnetic field according to :math:`H^{i+1/2,n+1/2}=H^{i+1/2,n-1/2}-\frac{1}{\mu_0}\frac{\Delta t}{\Delta z}\left[E^{i+1,n}-E^{i,n}\right]`

        :param pef: The prior electric field array (located half a time index away at :math:`n-1/2`)
        :param phf: The prior magnetic field array (located a whole time index away at :math:`n-1`)
        """
        # Create an array to hold the next field iteration
        nhfield = np.zeros(self._num_i, dtype=np.complex64)
        # Compute the values along the next field
        for i in range(self._num_i):
            term1 = phf[i]
            term2 = (self._delta_t*(pef[i+1]-pef[i]))/(self._vacuum_permeability*self._delta_z)
            nhfield[i] = term1 - term2
        self._hfield.append_field(nhfield)

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
    Represents either an electric or magnetic field.

    :param num_i: The number of spatial indexes in the field
    """

    def __init__(self, num_i):
        # Set the field time index to zero
        self._n = 0
        # Set the field spatial length to num_i
        self._num_i = num_i
        # Initialize a zero field
        self._field = list(np.zeros(num_i, dtype=np.complex64))

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
        self._n = n

    def get_field(self):
        """
        Gets the field at the current time index :math:`n`

        :return: The field at time `n`
        """
        return self._field[self._n]

    def __getitem__(self, key):
        """
        Allows the [] operator to be used to get field values
        """
        return self.get_index(key)

    def get_index(self, i):
        """
        Gets the value of the field at the current time index and at the the :math:`i` th spatial index. If the requested index is out of the field bounds, the returned value is zero.
        
        :param i: The spatial index of the field to access
        :return: The value of the field at time index :math:`n` and spatial index :math:`i`
        """
        # Check to see if the requested index is out of bounds, if so return zero
        if(i < 0 or i >= self._num_i):
            return np.complex64(0)

        # Return the requested field
        return (self._field[self._n])[i]

    def __setitem__(self, key, value):
        """
        Allows the [] operator to be used to set field values
        """
        return self.set_index(key, value)

    def set_index(self, i, value):
        """
        Sets the value of the field at the current time index and at the the :math:`i` th spatial index.
        
        :param i: The spatial index of the field to set
        :param value: The value to set at time index :math:`n` and spatial index :math:`i`
        """
        (self._field[self._n])[i] = np.complex64(value)

    def append_field(self, nfield):
        """
        Appends a new field to the list of fields, and updates the current time to that of the new field. Raises a ValueError if the new field is not of the correct spatial length.

        :param nfield: The new field to append of length num_i
        """
        # Check for nfield length, raise error if necessary
        if(len(nfield) != self._num_i):
            raise ValueError('The nfield argument is of the incorrect length, found ' + str(len(nfield)) + ', expected' + str(self._num_i))
        
        # Update time and append new field
        self._n = len(self._field)
        self._field.append(nfield)