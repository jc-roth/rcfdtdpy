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
    :param boundary: The boundary type of the field, either 'zero', for fields bounded by zeros, 'periodic' for periodic boundary conditions, or 'mirror' for boundaries that reflect inner field values.
    :param vacuum_permittivity: :math:`\epsilon_0`
    :param infinity_permittivity: :math:`\epsilon_\infty`
    :param vacuum_permeability: :math:`\mu_0`
    :param susceptibility: A susceptibility object
    :param initial_susceptibility: The initial susceptability. Eventually will be included in the susceptibility object.
    :param current_field: A field object that represents the current
    
    """
    
    def __init__(self, i0, i1, di, n0, n1, dn, cfield, boundary, vacuum_permittivity, infinity_permittivity, vacuum_permeability, susceptibility, initial_susceptibility):
        # Check that arguments have acceptable values
        if i0 > i1:
            raise ValueError("i0 must be less than or equal to i1.")
        elif n0 > n1:
            raise ValueError("n0 must be less than or equal to n1.")
        elif di <= 0:
            raise ValueError("di must be greater than zero.")
        elif dn <= 0:
            raise ValueError("dn must be greater than zero.")
        elif type(cfield) is not Field:
            raise TypeError("cfield must be of type Field.")
        # Save field dimensions and resolution
        self._i0 = i0
        self._i1 = i1
        self._di = di
        self._n0 = n0
        self._n1 = n1
        self._dn = dn
        # Determine the number of temporal and spatial cells in the field
        self._nlen, self._ilen = Sim.calc_dims(n0, n1, dn, i0, i1, di)
        # Create each field
        self._efield = Field(self._nlen, self._ilen, boundary)
        self._hfield = Field(self._nlen, self._ilen, boundary)
        # Save the current field
        self._cfield = cfield
        # Save constants
        self._epsilon0 = vacuum_permittivity
        self._epsiloninf = infinity_permittivity
        self._mu0 = vacuum_permeability
        self._chi = susceptibility
        self._chi0 = initial_susceptibility
        # Calculate simulation proportionality constants
        self._coeffe0 = self._epsiloninf/(self._epsiloninf + self._chi0)
        self._coeffe1 = 1.0/(self._epsiloninf + self._chi0)
        self._coeffe2 = self._dn/(self._epsilon0 * self._di * (self._epsiloninf + self._chi0))
        self._coeffe3 = self._dn/(self._epsilon0 * (self._epsiloninf + self._chi0))
        self._coeffh1 = self._dn/(self._mu0 * self._di)

    def __str__(self):
        """
        Returns a descriptive string of the Sim object.
        """
        to_return = ''
        to_return += '----------\nConstants:\n'
        to_return += 'epsilon0:' + '{0:.3f}'.format(self._epsilon0) + ' epsiloninf:' + '{0:.3f}'.format(self._epsiloninf) + ' mu0:' + '{0:.3f}'.format(self._mu0) + ' chi:' + '{0:.3f}'.format(self._chi) + ' chi0:' + '{0:.3f}'.format(self._chi0)
        to_return += '\n-------------\nCoefficients:\n'
        to_return += 'e1:' + '{0:.3f}'.format(self._coeffe0) + ' e2:' + '{0:.3f}'.format(self._coeffe1) + ' e3:' + '{0:.3f}'.format(self._coeffe2) + ' e4:' + '{0:.3f}'.format(self._coeffe3) + ' h2:' + '{0:.3f}'.format(self._coeffh1)
        to_return += '\n-------\nBounds:\n'
        to_return += 'i0:' + '{0:.3f}'.format(self._i0) + ' i1:' + '{0:.3f}'.format(self._i1) + ' di:' + '{0:.3f}'.format(self._di) + '\nn0:' + '{0:.3f}'.format(self._n0) + ' n1:' + '{0:.3f}'.format(self._n1) + ' dn:' + '{0:.3f}'.format(self._dn)
        to_return += '\n-------\nDimensions:\n'
        to_return += '( n x i ) ( ' + str(self._nlen) + ' x ' + str(self._ilen) + ' )'
        return to_return
        
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
            self._hfield.iterate(copy=True)
            self._efield.iterate(copy=True)
            self._cfield.iterate()
            

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
            term4 = self._coeffe3 * self._cfield[i]
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

    def _psi(self):
        """
        Calculates psi according to :math:`\psi^n=\sum^{n-1}_{m=0}E^{i,n-m}\Delta\chi_e^m` at the current time :math:`n` and position :math:`i`. Currently not implemented, and will simply return zero.

        :return: Zero
        """
        return 0

    def get_bound_res(self):
        """
        Returns the boundaries and resolution of the simulation

        :return: A tuple :code:`(n0, n1, dn, i0, i1, di)`
        """
        return (self._n0, self._n1, self._dn, self._i0, self._i1, self._di)

    def get_dims(self):
        """
        Returns the dimensions of the field in cells

        :returns: A tuple :code:`(nlen, ilen)` containing the temporal and spatial dimensions in cells
        """
        return (self._nlen, self._ilen)

    def export(self):
        """
        Exports all field values along with the spatial and temporal bounds of each field cell

        :return: A tuple :code:`(n, i, e, h, c)` where :code:`n` is a Numpy array containing the spatial bounds of each field cell, :code:`i` is a Numpy array containing the temporal bounds of each field cell, :code:`e` is a Numpy array containing the E-field (axis=0 is time and axis=1 is space), :code:`h` is a Numpy array containing the H-field (axis=0 is time and axis=1 is space), and :code:`c` is a Numpy array containing the current field (axis=0 is time and axis=1 is space)
        """
        # Calcualte the n and i arrays
        n = np.linspace(self._n0, self._n1, self._nlen, False)
        i = np.linspace(self._i0, self._i1, self._ilen, False)
        # Return
        return (n, i, self._efield.export(), self._hfield.export(), self._cfield.export())
        
    @staticmethod
    def calc_dims(n0, n1, dn, i0, i1, di):
        """
        Calculates the dimensions of the simulation in cells.

        :param i0: The spatial value at which the field starts
        :param i1: The spatial value at which the field ends
        :param di: The spatial step size
        :param n0: The temporal value at which the field starts
        :param n1: The temporal value at which the field ends
        :param dn: The temporal step size
        :return: A tuple (nlen, ilen) of the temporal and spatial dimensions
        """
        nlen = int(np.floor((n1-n0)/dn))
        ilen = int(np.floor((i1-i0)/di))
        return (nlen, ilen)

class Field:
    """
    Represents any field (i.e. electric, magnetic, current, susceptibility) using a 2D Numpy array. The zeroth axis represents increments in time and the first axis represents increments in space. Field dimensions are calculated via floor((i1-i0)/di) and floor((n1-n0)/dn).

    :param nlen: The number of temporal cells in the field
    :param ilen: The number of spatial cells in the field
    :param boundary: The boundary type of the field, either 'zero', for fields bounded by zeros, 'periodic' for periodic boundary conditions, or 'mirror' for boundaries that reflect inner field values.
    :param init: A Numpy array containing the field values to set
    """

    def __init__(self, nlen, ilen, boundary='zero', field=None):
        # Set the field time index to zero
        self._n = 0
        # Save the field dimensions
        self._nlen = nlen
        self._ilen = ilen
        # Save field boundary type
        self._boundary = boundary.lower()
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
        # Check to see if the requested index is out of bounds, if so return a value based on the boundary condition
        if(i < 0 or i >= self._ilen):
            if self._boundary == 'zero':
                return np.float32(0)
            elif self._boundary == 'periodic':
                return self._field[self._n, i % self._ilen]
            elif self._boundary == 'mirror':
                if i < 0:
                    return self._field[self._n, np.abs(i)]
                else:
                    return self._field[self._n, 2*self._ilen - (i + 1)]
        # Return the requested field
        return self._field[self._n, i]

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

    def iterate(self, copy=False):
        """
        Copies the field at the current temporal location to the next temporal location, then iterates the temporal location.
        """
        # Check for that n is within the accepted range
        if(self._n + 1 >= self._nlen):
            raise IndexError('Cannot iterate as the end of the temporal index has been reached.')
        if copy:
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