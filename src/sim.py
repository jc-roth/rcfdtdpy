import numpy as np
from tqdm import tqdm
"""
Contains the classes used to represent a simulation
"""

class Sim:
    """Represents a single simulation. Field is initialized to all zeros.

    :param i0: The spatial value at which the field starts
    :param i1: The spatial value at which the field ends
    :param di: The spatial step size
    :param n0: The temporal value at which the field starts
    :param n1: The temporal value at which the field ends
    :param dn: The temporal step size
    :param vacuum_permittivity: :math:`\epsilon_0`
    :param infinity_permittivity: :math:`\epsilon_\infty`
    :param vacuum_permeability: :math:`\mu_0`
    :param susceptibility: A susceptibility object
    :param initial_susceptibility: The initial susceptability. Eventually will be included in the susceptibility object.
    :param current_field: A field object that represents the current
    
    """
    
    def __init__(self, i0, i1, di, n0, n1, dn, vacuum_permittivity, infinity_permittivity, vacuum_permeability, susceptibility, initial_susceptibility, current_field):
        # Save constants
        self._epsilon0 = vacuum_permittivity
        self._epsiloninf = infinity_permittivity
        self._mu0 = vacuum_permeability
        self._chi = susceptibility
        self._chi0 = initial_susceptibility
        # Save current field
        self._cfield = current_field
        # Save field dimensions and resolution, create fields
        self._i0 = i0
        self._i1 = i1
        self._di = di
        self._n0 = n0
        self._n1 = n1
        self._dn = dn
        self._efield = Field(self._i0, self._i1, self._di, self._n0, self._n1, self._dn)
        self._hfield = Field(self._i0, self._i1, self._di, self._n0, self._n1, self._dn)
        # Determine the number of temporal and spatial cells in the field
        self._nlen, self._ilen = self._efield.dims()
        # Calculate simulation proportionality constants
        self._coeffe0 = self._epsiloninf/(self._epsiloninf + self._chi0)
        self._coeffe1 = 1.0/(self._epsiloninf + self._chi0)
        self._coeffe2 = self._dn/(self._epsilon0 * self._di * (self._epsiloninf + self._chi0))
        self._coeffe3 = self._dn/(self._epsilon0 * (self._epsiloninf + self._chi0))
        self._coeffh1 = self._dn/(self._mu0 * self._di)
        # Print coefficients
        print('Coefficients:\n=============')
        print(str(self._coeffe0)[:13])
        print(str(self._coeffe1)[:13])
        print(str(self._coeffe2)[:13])
        print(str(self._coeffe3)[:13])
        print(str(self._coeffh1)[:13])
        print('=============')

    def get_vacuum_permittivity(self):
        """
        Gets :math:`\epsilon_0`
        
        :returns: :math:`\epsilon_0`
        """
        return self._epsilon0

    def get_infinity_permittivity(self):
        """
        Gets :math:`\epsilon_\infty`
        
        :returns: :math:`\epsilon_\infty`
        """
        return self._epsiloninf

    def get_vacuum_permeability(self):
        """
        Gets :math:`\mu_0`
        
        :returns: :math:`\mu_0`
        """
        return self._mu0

    def get_delta_t(self):
        """
        Gets :math:`\Delta t`
        
        :returns: :math:`\Delta t`
        """
        return self._dn

    def get_delta_z(self):
        """
        Gets :math:`\Delta z`
        
        :returns: :math:`\Delta z`
        """
        return self._di

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

    def get_cfield(self):
        """
        Returns the current field as a Field object.

        :return: The current field as a Field object
        """
        return self._cfield
        
    def simulate(self):
        """
        Executes the simulation.
        """
        self._efield.set_time_index(0) # Set the time index of the electric field to zero
        self._hfield.set_time_index(0) # Set the time index of the magnetic field to zero
        self._cfield.set_time_index(0) # Set the time index of the current field to zero
        # Simulate for one less step than the number of temporal indicies because initializing the fields to zero takes up the first temporal index
        for n in tqdm(range(self._nlen-1)):
            # Calculate the H and E fields
            self._calc_hfield()
            self._calc_efield()
            # Iterate the H and E, and current fields
            self._hfield.iterate()
            self._efield.iterate()
            self._iterate_cfield()
            

    def _calc_efield(self):
        r"""
        Calcualtes the electric field according to :math:`E^{i,n+1}=\frac{\epsilon_\infty}{\epsilon_\infty+\chi_e^0}E^{i,n}+\frac{1}{\epsilon_\infty+\chi_e^0}\psi^n-\frac{1}{\epsilon_0\left[\epsilon_\infty+\chi_e^0\right]}\frac{\Delta t}{\Delta z}\left[H^{i+1/2,n+1/2}-H^{i-1/2,n+1/2}\right]-\frac{\Delta tI_f}{\epsilon_0\left[\epsilon_\infty+\chi_e^0\right]}`. Note that the prior electric field array is located half a time index away at :math:`n-1` and the prior magnetic field array is located half a time index away at :math:`n-1/2`.
        """
        # Compute the values along the field
        for i in range(self._ilen):
            # TODO Can this calculation be done via vectors? This will likely improve efficiency
            term1 = self._coeffe0 * self._efield[i]
            term2 = self._coeffe1 * self.psi()
            term3 = self._coeffe2 * (self._hfield[i]-self._hfield[i-1])
            term4 = self._coeffe3 * self._current(i)
            self._efield[i] = term1 + term2 - term3 - term4
        
    def _calc_hfield(self):
        r"""
        Calculates the magnetic field according to :math:`H^{i+1/2,n+1/2}=H^{i+1/2,n-1/2}-\frac{1}{\mu_0}\frac{\Delta t}{\Delta z}\left[E^{i+1,n}-E^{i,n}\right]`. Note that the prior electric field array is located half a time index away at :math:`n-1/2` and the prior magnetic field array is located a whole time index away at :math:`n-1`.
        """
        # Compute the values along the field
        for i in range(self._ilen):
            # TODO Can this calculation be done via vectors? This will likely improve efficiency
            term1 = self._hfield[i]
            term2 = self._coeffh1 * (self._efield[i+1]-self._efield[i])
            self._hfield[i] = term1 - term2
        
    def _iterate_cfield(self):
        """
        Iterates the current field by simply increasing the temporal index by one.
        """
        prior_time = self._cfield.get_time_index()
        self._cfield.set_time_index(prior_time+1)

    def psi(self):
        """
        Calculates psi according to :math:`\psi^n=\sum^{n-1}_{m=0}E^{i,n-m}\Delta\chi_e^m` at the current time :math:`n` and position :math:`i`. Currently not implemented, and will simply return zero.

        :return: Zero
        """
        return 0

    def _current(self, i):
        """
        Gets the current at location :math:`i` and current time :math:`n` using the simulation's associated current field.

        :return: The current at location :math:`i` and current time :math:`n`
        """
        return self._cfield[i]

class Field:
    """
    Represents any field (i.e. electric, magnetic, current, susceptibility) using a 2D Numpy array. The zeroth axis represents increments in time and the first axis represents increments in space. Field dimensions are calculated via floor((i1-i0)/di) and floor((n1-n0)/dn).

    :param i0: The spatial value at which the field starts
    :param i1: The spatial value at which the field ends
    :param di: The spatial step size
    :param n0: The temporal value at which the field starts
    :param n1: The temporal value at which the field ends
    :param dn: The temporal step size
    :param init: A Numpy array containing the field values to set

    """

    def __init__(self, i0, i1, di, n0, n1, dn, field=None):
        # Check that arguments have acceptable values
        if i0 > i1:
            raise ValueError("i0 must be less than or equal to i1.")
        elif n0 > n1:
            raise ValueError("n0 must be less than or equal to n1.")
        elif di <= 0:
            raise ValueError("di must be greater than zero.")
        elif dn <= 0:
            raise ValueError("dn must be greater than zero.")
        # Set the field time index to zero
        self._n = 0
        # Save the field dimensions and resolutions
        self._i0 = i0
        self._i1 = i1
        self._di = di
        self._n0 = n0
        self._n1 = n1
        self._dn = dn
        # Determine the number of temporal and spatial cells in the field
        self._nlen = int(np.floor((n1-n0)/dn))
        self._ilen = int(np.floor((i1-i0)/di))
        # Check to see if an initial field was provided
        if field is None:
            # Initialize zero-valued field
            self._field = np.zeros((self._nlen, self._ilen), dtype=np.float32)
        else:
            # Check that the given field has consistent dimensions
            nlen, ilen = np.shape(field)
            if(self._nlen != nlen or self._ilen != ilen):
                raise ValueError("The init field should have the same dimensions as those provided")
            # Initialize the given field
            self._field = field

    def dims(self):
        """
        Returns the dimensions of the field in cells

        :returns: A tuple :code:`(nlen, ilen)` containing the temporal and spatial dimensions in cells
        """
        return (self._nlen, self._ilen)

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
        if(n < 0 or n >= self._nlen):
            raise IndexError('The n argument is of out of bounds')
        self._n = n

    def get_field(self, n=-1):
        """
        Gets the field spatial values as well as amplitude at time index :math:`n` or the current time if :math:`n` is unspecified.

        :param n: The time index :math:`n`
        :return: A tuple :code:`(z, f)` containing the spatial coordinates of the cells in the field and the field amplitudes at each corresponding cell
        """
        # Calcualte the z array
        z = np.linspace(self._i0, self._i1, self._ilen, False)
        # If n is -1, return the current field
        if(n == -1):
            return (z, self._field[self._n])
        # Check for that n is within the accepted range
        if(n < 0 or n >= self._nlen):
            raise IndexError('The n argument is of out of bounds')
        return (z, self._field[n])

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
        if(i < 0 or i >= self._ilen):
            return np.float32(0)
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
        self._field[self._n,i] = np.float32(value)

    def iterate(self):
        """
        Copies the field at the current temporal location to the next temporal location, then iterates the temporal location.
        """
        # Check for that n is within the accepted range
        if(self._n + 1 >= self._nlen):
            raise IndexError('Cannot iterate as the end of the temporal index has been reached.')
        # Copy the current field to the next temporal index
        self._field[self._n+1] = self._field[self._n]
        # Iterate the temporal index
        self._n += 1

    def export(self):
        """
        Returns the Numpy array that contains the temporal (axis=0) and spatial values (axis=1) of the field.

        :return: A Numpy array
        """
        return self._field