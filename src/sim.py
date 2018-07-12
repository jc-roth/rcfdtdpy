import numpy as np
from tqdm import tqdm
"""
Contains the classes used to represent a simulation
"""

class Sim:
    r"""Represents a single simulation. Field is initialized to all zeros.

    :param i0: The spatial value at which the field starts
    :param i1: The spatial value at which the field ends
    :param di: The spatial step size
    :param n0: The temporal value at which the field starts
    :param n1: The temporal value at which the field ends
    :param dn: The temporal step size
    :param epsilon0: :math:`\epsilon_0`
    :param mu0: :math:`\mu_0`
    :param cfield: The current field
    :param boundary: The boundary type of the field, either 'zero', for fields bounded by zeros, 'periodic' for periodic boundary conditions, or 'mirror' for boundaries that reflect inner field values
    :param mat: A Mat object or a list of Mat objects that represent the materials present in the simulation
    :param dtype: The data type to store the field values in
    :param nstore: The number of temporal steps to save, defaults to zero
    :param storelocs: A list of locations to save field values of at each step in time
    """
    
    def __init__(self, i0, i1, di, n0, n1, dn, epsilon0, mu0, cfield, boundary, mat, dtype=np.complex64, nstore=0, storelocs = []):
        # -------------
        # INITIAL SETUP
        # -------------
        # Check that arguments have acceptable values
        if i0 > i1:
            raise ValueError("i0 must be less than or equal to i1")
        elif n0 > n1:
            raise ValueError("n0 must be less than or equal to n1")
        elif di <= 0:
            raise ValueError("di must be greater than zero")
        elif dn <= 0:
            raise ValueError("dn must be greater than zero")
        elif type(cfield) is not np.ndarray:
            raise TypeError("cfield must be of type numpy.ndarray")
        elif type(mat) != Mat or (type(mat) is list and type(mat[0]) != Mat):
            raise TypeError("mat must be either a Mat object or a list of Mat objects")
        # Determine the number of temporal and spatial cells in the field
        self._nlen, self._ilen = Sim.calc_dims(n0, n1, dn, i0, i1, di)
        # Raise further errors
        if (self._nlen, self._ilen) != np.shape(cfield):
            raise ValueError("Expected cfield to have dimensions " + str(self.get_dims()) + " but found " + str(np.shape(cfield)) + " instead")
        elif nstore > self._nlen:
            raise ValueError("nstore=" + str(nstore) + ", cannot be greater than nlen=" + str(self._nlen))
        # Save data type
        self._dtype = dtype
        # Save field dimensions and resolution
        self._i0 = i0
        self._i1 = i1
        self._di = di
        self._n0 = n0
        self._n1 = n1
        self._dn = dn
        # --------------
        # MATERIAL SETUP
        # --------------
        # Put the mat variable into a list if it isn't already
        if type(mat) == Mat:
            mat = [mat]
        # Save the material
        self._mats = mat
        # Check to see if there is any material overlap
        self._matpos = np.zeros(self._ilen)
        for m in mat:
            self._matpos = np.add(self._matpos, m.get_pos())
        # Raise error if there is overlap
        if np.max(self._matpos) > 1:
            raise ValueError("Found overlap between materials, remove before proceeding.")
        # --------------
        # BOUNDARY SETUP
        # --------------
        # Setup boundary condition
        self._bound = boundary
        if self._bound == 'absorbing':
            self._eprev0 = np.complex64(0)
            self._eprev1 = np.complex64(0)
            self._erprev0 = np.complex64(0)
            self._erprev1 = np.complex64(0)
        # -----------
        # FIELD SETUP
        # -----------
        # Create each field
        self._efield = np.zeros(self._ilen, dtype=self._dtype)
        self._hfield = np.zeros(self._ilen, dtype=self._dtype)
        # Create each reference field
        self._efieldr = np.zeros(self._ilen, dtype=self._dtype)
        self._hfieldr = np.zeros(self._ilen, dtype=self._dtype)
        # -------------------
        # STORED VALUES SETUP
        # -------------------
        # Determine how often to save the field and arrays to save to
        self._nstore = nstore
        # Check to see if any stores are requested
        if self._nstore == 0:
            # Never store
            self._n_step_btwn_store = -1
        else:
            # Store
            self._n_step_btwn_store = int(np.ceil(self._nlen/self._nstore))
            self._efield_save = np.zeros((self._nstore, self._ilen), dtype=self._dtype)
            self._hfield_save = np.zeros((self._nstore, self._ilen), dtype=self._dtype)
            self._efieldr_save = np.zeros((self._nstore, self._ilen), dtype=self._dtype)
            self._hfieldr_save = np.zeros((self._nstore, self._ilen), dtype=self._dtype)
        # Save storeloc info
        self._storelocs = storelocs
        self._nlocs = len(self._storelocs)
        self._storeind = list(range(self._nlocs))
        # Check to see if any storelocs are requested
        if self._nlocs != 0:
            # Create arrays to store the field values in each location
            self._nlocs = len(storelocs)
            self._efield_locs = np.zeros((self._nlen, self._nlocs), dtype=self._dtype)
            self._hfield_locs = np.zeros((self._nlen, self._nlocs), dtype=self._dtype)
            self._efieldr_locs = np.zeros((self._nlen, self._nlocs), dtype=self._dtype)
            self._hfieldr_locs = np.zeros((self._nlen, self._nlocs), dtype=self._dtype)
        # -------------
        # CURRENT SETUP
        # -------------
        # Save the current field
        self._cfield = cfield
        # ---------------
        # CONSTANTS SETUP
        # ---------------
        # Save constants
        self._epsilon0 = epsilon0
        self._mu0 = mu0
        # Sum the epsiloninf values of each material to get the final epsiloninf array (note there is no material overlap)
        self._epsiloninf = np.zeros(self._ilen, dtype=self._dtype)
        for m in mat:
            self._epsiloninf = np.add(self._epsiloninf, m.get_epsiloninf())
        # Sum the chi0 values of each material to get the final epsiloninf array (note there is no material overlap)
        self._chi0 = np.zeros(self._ilen, dtype=self._dtype)
        for m in mat:
            self._chi0 = np.add(self._chi0, m.get_chi0())
        # Calculate simulation proportionality constants
        self._coeffe0 = self._epsiloninf/(self._epsiloninf + self._chi0)
        self._coeffe1 = 1.0/(self._epsiloninf + self._chi0)
        self._coeffe2 = self._dn/(self._epsilon0 * self._di * (self._epsiloninf + self._chi0))
        self._coeffe3 = self._dn/(self._epsilon0 * (self._epsiloninf + self._chi0))
        self._coeffh1 = self._dn/(self._mu0 * self._di)
        # Create simulation reference proportionality constants (the reference sees chi0=0 and epsiloninf=1)
        self._coeffe0r = np.complex64(1)
        self._coeffe2r = self._dn/(self._epsilon0 * self._di)
        self._coeffe3r = self._dn/(self._epsilon0)
        self._coeffh1r = self._dn/(self._mu0 * self._di)

    def __str__(self):
        """
        Returns a descriptive string of the Sim object.
        """
        to_return = ''
        to_return += '----------\nConstants:\n'
        to_return += 'epsilon0:' + '{0:.3f}'.format(self._epsilon0) + ' epsiloninf:' + '{0:.3f}'.format(self._epsiloninf) + ' mu0:' + '{0:.3f}'.format(self._mu0) + ' chi:' + 'NOT YET IMPLEMENTED' + ' chi0:' + '{0:.3f}'.format(self._chi0)
        to_return += '\n-------------\nCoefficients:\n'
        to_return += 'e1:' + '{0:.3f}'.format(self._coeffe0) + ' e2:' + '{0:.3f}'.format(self._coeffe1) + ' e3:' + '{0:.3f}'.format(self._coeffe2) + ' e4:' + '{0:.3f}'.format(self._coeffe3) + ' h2:' + '{0:.3f}'.format(self._coeffh1)
        to_return += '\n-------\nBounds:\n'
        to_return += 'i0:' + '{0:.3f}'.format(self._i0) + ' i1:' + '{0:.3f}'.format(self._i1) + ' di:' + '{0:.3f}'.format(self._di) + '\nn0:' + '{0:.3f}'.format(self._n0) + ' n1:' + '{0:.3f}'.format(self._n1) + ' dn:' + '{0:.3f}'.format(self._dn)
        to_return += '\n-------\nDimensions:\n'
        to_return += '( n x i ): ( ' + str(self._nlen) + ' x ' + str(self._ilen) + ' )'
        return to_return
        
    def simulate(self):
        """
        Executes the simulation.
        """
        n_save = 0
        # Simulate for one less step than the number of temporal indicies because initializing the fields to zero takes up the first temporal index
        for n in tqdm(range(self._nlen), desc='Executing simulation'):
            # Calculate the H and E fields
            if self._bound == 'zero': # Zero boundary condition
                self._zero(n)
            if self._bound == 'absorbing': # Absorbing boundary condition
                self._absorbing(n)
            # Save the new fields if storing is on (self._n_step_btwn_store != -1) and at an appropriate step
            if self._n_step_btwn_store != -1 and n % self._n_step_btwn_store == 0:
                self._hfield_save[n_save] = self._hfield
                self._efield_save[n_save] = self._efield
                self._hfieldr_save[n_save] = self._hfieldr
                self._efieldr_save[n_save] = self._efieldr
                n_save += 1
            # Save specific field locations if storing has been requested
            if self._nlocs != 0:
                # Store each location
                self._efield_locs[n, self._storeind] = self._efield[self._storelocs]
                self._efieldr_locs[n, self._storeind] = self._efieldr[self._storelocs]
                self._hfield_locs[n, self._storeind] = self._hfield[self._storelocs]
                self._hfieldr_locs[n, self._storeind] = self._hfieldr[self._storelocs]

    def _absorbing(self, n):
        """
        Computes the E-field and H-fields at time step n with absorbing boundaries
        """
        # Update Psi
        self._update_psi()
        # Compute H-field and update
        self._update_hfield(n)
        self._update_hfieldr(n)
        # Compute E-field and update
        self._update_efield(n)
        self._update_efieldr(n)
        # Set the field values at the boundary to the previous value one away from the boundary, this somehow results in absorption, I'm not really sure how... I think it has something to do with preventing any wave reflection, meaning that the field values just end up going to zero. It would be a good idea to ask Ben about this.
        self._efield[0] = self._eprev0
        self._efield[-1] = self._eprev1
        self._efieldr[0] = self._erprev0
        self._efieldr[-1] = self._erprev1
        # Save the field values one away from each boundary for use next iteration
        self._eprev0 = self._efield[1]
        self._eprev1 = self._efield[-2]
        self._erprev0 = self._efieldr[1]
        self._erprev1 = self._efieldr[-2]

    def _zero(self, n):
        """
        Computes the E-field and H-fields at time step n with constant zero boundaries.
        """
        # Update Psi
        self._update_psi()
        # Compute H-field and update
        self._update_hfield(n)
        self._update_hfieldr(n)
        # Compute E-field and update
        self._update_efield(n)
        self._update_efieldr(n)
        
    def _update_hfield(self, n):
        """
        Updates the H-field to the values at the next iteration
        """
        h_t1 = self._hfield[:-1]
        h_t2 = self._coeffh1 * (self._efield[1:]-self._efield[:-1])
        self._hfield[:-1] = h_t1 - h_t2

    def _update_efield(self, n):
        """
        Updates the E-field to the values at the next iteration
        """
        e_t1 = self._coeffe0[1:] * self._efield[1:]
        e_t2 = self._coeffe1[1:] * self._compute_psi()[1:]
        e_t3 = self._coeffe2[1:] * (self._hfield[1:]-self._hfield[:-1])
        e_t4 = self._coeffe3[1:] * self._cfield[n,1:]
        self._efield[1:] = e_t1 + e_t2 - e_t3 - e_t4
        
    def _update_hfieldr(self, n):
        """
        Updates the reference H-field to the values at the next iteration
        """
        h_t1 = self._hfieldr[:-1]
        h_t2 = self._coeffh1r * (self._efieldr[1:]-self._efieldr[:-1])
        self._hfieldr[:-1] = h_t1 - h_t2

    def _update_efieldr(self, n):
        """
        Updates the reference E-field to the values at the next iteration
        """
        e_t1 = self._coeffe0r * self._efieldr[1:]
        e_t3 = self._coeffe2r * (self._hfieldr[1:]-self._hfieldr[:-1])
        e_t4 = self._coeffe3r * self._cfield[n,1:]
        self._efieldr[1:] = e_t1 - e_t3 - e_t4

    def _compute_psi(self):
        """
        Calculates psi at all points in the simulation using all materials in the simulation
        """
        # Create an array to hold psi
        psi = np.zeros(self._ilen)
        for m in self._mats:
            psi = np.add(psi, m.compute_psi())
        # Return
        return psi

    def _update_psi(self):
        """
        Updates the value of psi_1 and psi_2 in each material in the simulation
        """
        # Iterate through all materials and update each
        for m in self._mats:
            m.update_psi(self._efield)

    def get_bound_res(self):
        """
        Returns the boundaries and resolution of the simulation.

        :return: A tuple :code:`(n0, n1, dn, i0, i1, di)`
        """
        return (self._n0, self._n1, self._dn, self._i0, self._i1, self._di)

    def get_dims(self):
        """
        Returns the dimensions of the field in cells.

        :returns: A tuple :code:`(nlen, ilen)` containing the temporal and spatial dimensions in cells
        """
        return (self._nlen, self._ilen)

    def export(self):
        """
        Exports all field values along with the spatial and temporal bounds of each field cell

        :return: A tuple :code:`(n, i, e, h, er, hr, ls, els, erls, hls, hrls, c)` where :code:`n` is a Numpy array containing the spatial bounds of each field cell, :code:`i` is a Numpy array containing the temporal bounds of each field cell, :code:`e` is a Numpy array containing the E-field (axis=0 is time and axis=1 is space), :code:`h` is a Numpy array containing the H-field (axis=0 is time and axis=1 is space), :code:`er` is a Numpy array containing the reference E-field (axis=0 is time and axis=1 is space), :code:`hr` is a Numpy array containing the reference H-field (axis=0 is time and axis=1 is space), :code:`ls` is the list :code:`storelocs` (the same :code:`storelocs` that is passed to the Sim class during instantiation), :code:`els` is a Numpy array containing the E-field at storelocs at each point in time (axis=0 is time and axis=1 is the respective storeloc location), :code:`erls` is a Numpy array containing the reference E-field at storelocs at each point in time (axis=0 is time and axis=1 is the respective storeloc location), :code:`hls` is a Numpy array containing the H-field at storelocs at each point in time (axis=0 is time and axis=1 is the respective storeloc location), :code:`hrls` is a Numpy array containing the reference H-field at storelocs at each point in time (axis=0 is time and axis=1 is the respective storeloc location), and :code:`c` is a Numpy array containing the current field (axis=0 is time and axis=1 is space)
        """
        # Calcualte the n and i arrays
        n = np.linspace(self._n0, self._n1, self._nlen, False)
        i = np.linspace(self._i0, self._i1, self._ilen, False)
        # Check to see what was stored
        if self._nstore == 0:
            self._efield_save = None
            self._hfield_save = None
            self._efieldr_save = None
            self._hfieldr_save = None
        if self._nlocs == 0:
            self._efield_locs = None
            self._efieldr_locs = None
            self._hfield_locs = None
            self._hfieldr_locs = None
        # Return
        return (n, i, self._efield_save, self._hfield_save, self._efieldr_save, self._hfieldr_save, self._storelocs,  self._efield_locs, self._efieldr_locs, self._hfield_locs, self._hfieldr_locs, self._cfield)
        
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
        ilen = Sim._calc_idim(i0, i1, di)
        return (nlen, ilen)
        
    @staticmethod
    def _calc_idim(i0, i1, di):
        """
        Calculates the dimensions of the simulation in cells.

        :param i0: The spatial value at which the field starts
        :param i1: The spatial value at which the field ends
        :param di: The spatial step size
        :return: The spatial dimension
        """
        return int(np.floor((i1-i0)/di)+2) # Add two to account for boundary conditions
        
    @staticmethod
    def calc_mat_dims(i0, i1, di, mat0, mat1):
        """
        Calculates the dimensions of a material.

        :param i0: The spatial value at which the simulation starts
        :param i1: The spatial value at which the simulation ends
        :param di: The spatial step size
        :param mat0: The spatial value at which the material starts
        :param mat1: The spatial value at which the material ends
        :return: A tuple :code:`(matstart, matlen)` containing the index of the simulation at which the material starts and the number of indicies that the material spans
        """
        ilen = Sim._calc_idim(i0, i1, di)
         # Calculate the length of the material
        matlen = Sim._calc_idim(mat0, mat1, di)
         # Calculate the start point of the material
        matstart = int(np.floor((mat0-i0)/di))
        # Return
        return (matstart, matlen)

class Mat:
    r"""
    The Mat class is used to represent a material present in a simulation

    :param dn: The temporal step size
    :param ilen: The number of temporal indicies in the simulation
    :param mat0: The starting index of the material
    :param epsiloninf: :math:`\epsilon_\infty` inside the material
    :param mata1: A matrix representing :math:`A_1` where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial index
    :param mata2: A matrix representing :math:`A_2` where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial index
    :param matg: A matrix representing :math:`\gamma` where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial index
    :param matb: A matrix representing :math:`\beta` where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial index
    """

    def __init__(self, dn, ilen, mat0, epsiloninf, mata1, mata2, matg, matb, dtype=np.complex64):
        # -------------
        # INITIAL SETUP
        # -------------
        # Check for error
        if np.shape(mata1) != np.shape(mata2) or np.shape(mata1) != np.shape(matg) or np.shape(mata1) != np.shape(matb):
            raise ValueError("The dimensions of mata1, mata2, matg, and matb should be the same")
        # Save arguments
        self._dn = dn
        self._ilen = ilen
        self._mat0 = mat0
        self._dtype = dtype
        # Get and save material dimension info
        if len(np.shape(mata1)) > 1:
            self._jlen = np.shape(mata1)[0]
            self._matlen = np.shape(mata1)[1]
        else:
            self._jlen = 1
            self._matlen = np.shape(mata1)[0]
        # Check for error
        if self._mat0 < 0 or self._mat0 + self._matlen > self._ilen:
            raise ValueError("Material cannot start at i=" + str(self._mat0) + " and end at i=" + str(self._mat0 + self._matlen) + " as this exceeds the dimensions of the simulation.")
        # Reshape arrays so that they can be indexed correctly
        mata1 = np.reshape(mata1, (self._jlen, self._matlen))
        mata2 = np.reshape(mata2, (self._jlen, self._matlen))
        matg = np.reshape(matg, (self._jlen, self._matlen))
        matb = np.reshape(matb, (self._jlen, self._matlen))
        # Epsilon_infinity is equal to one in vacuum, so only set self._epsiloninf equal to epsiloninf in the material
        epsiloninf_repeat = np.repeat(epsiloninf, self._matlen)
        self._epsiloninf = np.pad(epsiloninf_repeat, (self._mat0, self._ilen - (self._mat0 + self._matlen)), 'constant', constant_values=1)
        # --------------
        # MATERIAL SETUP
        # --------------
        # Calculate susceptability beta and gamma sums and exponents
        b_min_g = np.add(matb, -matg)
        min_b_min_g = np.add(-matb, -matg)
        self._exp_1 = np.exp(np.multiply(b_min_g, self._dn))
        self._exp_2 = np.exp(np.multiply(min_b_min_g, self._dn))
        # Calculate initial susceptability values
        chi0_1 = np.zeros((self._jlen, self._matlen), dtype=self._dtype) # Set chi0_1=0 initially
        chi0_2 = np.zeros((self._jlen, self._matlen), dtype=self._dtype) # Set chi0_2=0 initially
        for j in range(self._jlen):
            for mi in range(self._matlen):
                if np.abs(b_min_g[j, mi]) > 1e-8:
                    # If beta-gamma is not small (i.e. if omega!=0), then calculate chi0_1, otherwise do not calculate as divide by zero error will be thrown
                    chi0_1[j, mi] = np.multiply(np.divide(mata1[j, mi], b_min_g[j, mi]), np.subtract(self._exp_1[j, mi], 1))
                    # Calculate chi0_2 normally
                    chi0_2[j, mi] = np.multiply(np.divide(mata2[j, mi], min_b_min_g[j, mi]), np.subtract(self._exp_2[j, mi], 1))
                else:
                    # If beta-gamma is small, multiply chi0_2 by negative one, not sure why, just taking from Ben
                    chi0_2[j, mi] = np.multiply(np.divide(-mata2[j, mi], min_b_min_g[j, mi]), np.subtract(self._exp_2[j, mi], 1))
        # Calclate first delta susceptabiility values
        self._dchi0_1 = np.multiply(chi0_1, np.subtract(1, self._exp_1))
        self._dchi0_2 = np.multiply(chi0_2, np.subtract(1, self._exp_2))
        # Initialize psi values to zero
        self._psi_1 = np.zeros((self._jlen, self._matlen), dtype=self._dtype)
        self._psi_2 = np.zeros((self._jlen, self._matlen), dtype=self._dtype)
        # Calculate chi0
        chi0_j = np.add(chi0_1, chi0_2)
        chi0 = np.sum(chi0_j, axis=0)
        # Pad chi0 so that it spans the length of the simulation
        chi0_padded = np.pad(chi0, (self._mat0, self._ilen - (self._mat0 + self._matlen)), 'constant')
        self._chi0 = chi0_padded

    def get_pos(self):
        r"""
        Returns a Numpy array of value 0 outside of the material and 1 inside of the material.

        :return: A Numpy array of value 0 outside of the material and 1 inside of the material
        """
        return np.pad(np.repeat(1, self._matlen), (self._mat0, self._ilen - (self._mat0 + self._matlen)), 'constant')

    def get_epsiloninf(self):
        r"""
        Returns the high frequency susceptability of the material :math:`\epsilon_\infty`.

        :return: A Numpy array of length :code:`ilen` of value 1 outside of the material and :math:`\epsilon_\infty` inside of the material
        """
        return self._epsiloninf

    def get_chi0(self):
        r"""
        Returns the initial susceptibility of the material :math:`\chi_0`.

        :return: A Numpy array of length :code:`ilen` of value 0 outside of the material and :math:`\chi_0` inside of the material
        """
        return self._chi0

    def compute_psi(self):
        """
        Calculates psi at all points in the simulation using the current value of psi_1 and psi_2
        """
        # Find the psi matrix
        psi_j = np.add(self._psi_1, self._psi_2)
        # Sum the psi matrix along axis=0 to combine all oscillators
        psi = np.sum(psi_j, axis=0)
        # Pad the psi array so that it spans the length of the simulation
        psi_padded = np.pad(psi, (self._mat0, self._ilen - (self._mat0 + self._matlen)), 'constant')
        # Return
        return psi_padded

    def update_psi(self, efield):
        """
        Updates the value of psi_1 and psi_2.

        :param efield: The efield to use in update calculations
        """
        # Copy the efield so that instead of being a vector it is a matrix composed of horizontal efield vectors
        e = np.tile(efield[self._mat0:self._mat0 + self._matlen], (self._jlen, 1))
        # Calculate first term
        t1_1 = np.multiply(e, self._dchi0_1)
        t1_2 = np.multiply(e, self._dchi0_2)
        # Calculate second term
        t2_1 = np.multiply(self._psi_1, self._exp_1)
        t2_2 = np.multiply(self._psi_2, self._exp_2)
        # Update next psi values
        self._psi_1 = np.add(t1_1, t2_1)
        self._psi_2 = np.add(t1_2, t2_2)