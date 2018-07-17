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
    :param boundary: The boundary type of the field, either 'zero', for fields bounded by zeros, 'periodic' for periodic boundary conditions, or 'mirror' for boundaries that reflect inner field values
    :param current: A Current object or a list of Current objects that represent the crrents present in the simulation, defaults to none
    :param mat: A Mat object or a list of Mat objects that represent the materials present in the simulation, defaults to none
    :param nstore: The number of temporal steps to save, defaults to zero
    :param storelocs: A list of locations to save field values of at each step in time
    :param dtype: The data type to store the field values in
    """
    
    def __init__(self, i0, i1, di, n0, n1, dn, epsilon0, mu0, boundary, current=[], mat=[], nstore=0, storelocs = [], dtype=np.complex64):
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
        elif (not ((type(mat) == Mat) or (type(mat) is list))) or ((type(mat) is list) and (len(mat) != 0) and (type(mat[0]) != Mat)):
            raise TypeError("mat must be either a Mat object or a list of Mat objects")
        elif (not ((type(current) == Current) or (type(current) is list))) or ((type(current) is list) and (len(current) != 0) and (type(current[0]) != Current)):
            raise TypeError("current must be either a Current object or a list of Current objects")
        # Determine the number of temporal and spatial cells in the field
        self._nlen, self._ilen = Sim.calc_dims(n0, n1, dn, i0, i1, di)
        # Raise further errors
        if nstore > self._nlen:
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
        # -------------
        # CURRENT SETUP
        # -------------
        # Put the current variable into a list if it isn't already
        if type(current) == Current:
            current = [current]
        # List already exits, create an empty current
        elif len(current) == 0:
            # Create an empty current
            c = np.zeros((1, 1))
            current.append(Current(self._nlen, self._ilen, 0, 0, c, dtype=self._dtype))
        # Save the currents
        self._currents = current
        # --------------
        # MATERIAL SETUP
        # --------------
        # Put the mat variable into a list if it isn't already
        if type(mat) == Mat:
            mat = [mat]
        # List already exits, create an uninteracting material
        elif len(mat) == 0:
            # Create an empty material
            m = np.ones((1, 1))
            mat.append(Mat(self._dn, self._ilen, 0, 1, m*0, m*0, m, m))
        # Save the material
        self._mats = mat
        # Check to see if there is any material overlap
        self._matpos = np.zeros(self._ilen)
        for m in self._mats:
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
        # Check to see if any storelocs are requested
        if self._nlocs != 0:
            # Create arrays to store the field values in each location
            self._nlocs = len(storelocs)
            self._efield_locs = np.zeros((self._nlen, self._nlocs), dtype=self._dtype)
            self._hfield_locs = np.zeros((self._nlen, self._nlocs), dtype=self._dtype)
            self._efieldr_locs = np.zeros((self._nlen, self._nlocs), dtype=self._dtype)
            self._hfieldr_locs = np.zeros((self._nlen, self._nlocs), dtype=self._dtype)
        # ---------------
        # CONSTANTS SETUP
        # ---------------
        # Save constants
        self._epsilon0 = epsilon0
        self._mu0 = mu0
        # Sum the epsiloninf values of each material to get the final epsiloninf array (note there is no material overlap)
        self._epsiloninf = np.zeros(self._ilen, dtype=self._dtype)
        for m in mat:
            self._epsiloninf = np.add(self._epsiloninf, m._get_epsiloninf())
        # Sum the chi0 values of each material to get the final epsiloninf array (note there is no material overlap)
        self._chi0 = np.zeros(self._ilen, dtype=self._dtype)
        for m in mat:
            self._chi0 = np.add(self._chi0, m._get_chi0())
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
                self._efield_locs[n,:] = self._efield[self._storelocs]
                self._efieldr_locs[n,:] = self._efieldr[self._storelocs]
                self._hfield_locs[n,:] = self._hfield[self._storelocs]
                self._hfieldr_locs[n,:] = self._hfieldr[self._storelocs]

    def _absorbing(self, n):
        """
        Computes the E-field and H-fields at time step n with absorbing boundaries.
        """
        # Update Psi
        self._update_mat(n)
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
        self._update_mat(n)
        # Compute H-field and update
        self._update_hfield(n)
        self._update_hfieldr(n)
        # Compute E-field and update
        self._update_efield(n)
        self._update_efieldr(n)
        
    def _update_hfield(self, n):
        """
        Updates the H-field to the values at the next iteration. Should be called once per simulation step.
        """
        h_t1 = self._hfield[:-1]
        h_t2 = self._coeffh1 * (self._efield[1:]-self._efield[:-1])
        self._hfield[:-1] = h_t1 - h_t2

    def _update_efield(self, n):
        """
        Updates the E-field to the values at the next iteration. Should be called once per simulation step.
        """
        e_t1 = self._coeffe0[1:] * self._efield[1:]
        e_t2 = self._coeffe1[1:] * self._compute_psi()[1:]
        e_t3 = self._coeffe2[1:] * (self._hfield[1:]-self._hfield[:-1])
        e_t4 = self._coeffe3[1:] * self._get_current(n)[1:]
        self._efield[1:] = e_t1 + e_t2 - e_t3 - e_t4
        
    def _update_hfieldr(self, n):
        """
        Updates the reference H-field to the values at the next iteration. Should be called once per simulation step.
        """
        h_t1 = self._hfieldr[:-1]
        h_t2 = self._coeffh1r * (self._efieldr[1:]-self._efieldr[:-1])
        self._hfieldr[:-1] = h_t1 - h_t2

    def _update_efieldr(self, n):
        """
        Updates the reference E-field to the values at the next iteration. Should be called once per simulation step.
        """
        e_t1 = self._coeffe0r * self._efieldr[1:]
        e_t3 = self._coeffe2r * (self._hfieldr[1:]-self._hfieldr[:-1])
        e_t4 = self._coeffe3r * self._get_current(n)[1:]
        self._efieldr[1:] = e_t1 - e_t3 - e_t4

    def _get_current(self, n):
        """
        Calculates the current at all points in the simulation using all the currents in the simulation

        :param n: The temporal index :math:`n` to calculate the current at
        """
        # Create an array to hold the current
        current = np.zeros(self._ilen)
        for c in self._currents:
            current = np.add(current, c._get_current(n))
        # Return
        return current

    def _compute_psi(self):
        """
        Calculates psi at all points in the simulation using all materials in the simulation.
        """
        # Create an array to hold psi
        psi = np.zeros(self._ilen)
        for m in self._mats:
            psi = np.add(psi, m._compute_psi())
        # Return
        return psi

    def _update_mat(self, n):
        """
        Updates the each material in the simulation using the `_update_mat` function. Should be called once per simulation step.
        """
        # Iterate through all materials and update each
        for m in self._mats:
            m._update_mat(n, self._efield)

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

    def get_mats(self):
        """
        Returns the list of Mat objects present in the simulation.

        :returns: A list of Mat objects
        """
        return self._mats

    def export_fields(self):
        """
        Exports all stored field values (the number of which is determined by :code:`nstore` at initialization) along with the spatial and temporal bounds of each field cell.

        :return: A tuple :code:`(n, i, e, h, er, hr)` where :code:`n` is a Numpy array containing the spatial bounds of each field cell, :code:`i` is a Numpy array containing the temporal bounds of each field cell, :code:`e` is a Numpy array containing the E-field (axis=0 is time and axis=1 is space), :code:`h` is a Numpy array containing the H-field (axis=0 is time and axis=1 is space), :code:`er` is a Numpy array containing the reference E-field (axis=0 is time and axis=1 is space), :code:`hr` is a Numpy array containing the reference H-field (axis=0 is time and axis=1 is space)
        """
        # Calcualte the n and i arrays
        n = np.linspace(self._n0, self._n1, self._nstore, False)
        i = np.linspace(self._i0, self._i1, self._ilen, False)
        # Check to see what was stored
        if self._nstore == 0:
            self._efield_save = None
            self._hfield_save = None
            self._efieldr_save = None
            self._hfieldr_save = None
        # Return
        return (n, i, self._efield_save, self._hfield_save, self._efieldr_save, self._hfieldr_save)

    def export_locs(self):
        """
        Exports the value of each field at a specific location(s) (specified with :code:`storelocs` at initialization) at each point in time.

        :return: A tuple :code:`(n, ls, els, erls, hls, hrls)` where :code:`n` is a Numpy array containing the spatial bounds of each field cell, :code:`ls` is the list :code:`storelocs` (the same :code:`storelocs` that is passed to the Sim class during instantiation), :code:`els` is a Numpy array containing the E-field at storelocs at each point in time (axis=0 is time and axis=1 is the respective storeloc location), :code:`erls` is a Numpy array containing the reference E-field at storelocs at each point in time (axis=0 is time and axis=1 is the respective storeloc location), :code:`hls` is a Numpy array containing the H-field at storelocs at each point in time (axis=0 is time and axis=1 is the respective storeloc location), and :code:`hrls` is a Numpy array containing the reference H-field at storelocs at each point in time (axis=0 is time and axis=1 is the respective storeloc location)
        """
        # Calcualte the n array
        n = np.linspace(self._n0, self._n1, self._nlen, False)
        # Check to see if the value of each field at a specific location was saved over time
        if self._nlocs == 0:
            self._efield_locs = None
            self._efieldr_locs = None
            self._hfield_locs = None
            self._hfieldr_locs = None
        # Return
        return (n, self._storelocs,  self._efield_locs, self._efieldr_locs, self._hfield_locs, self._hfieldr_locs)

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
        ilen = int(np.floor((i1-i0)/di)+2) # Add two to account for boundary conditions
        return (nlen, ilen)


class Current:
    r"""
    The Current class is used to represent a current in the simulation.

    :param nlen: The number of temporal indicies in the simulation
    :param ilen: The number of spatial indicies in the simulation
    :param n0: The starting temporal index of the current
    :param i0: The starting spatial index of the current
    :param current: A matrix representing the current, where axis=0 represents locations in time :math:`n` and axis=1 represents locations in space :math:`i`
    :param dtype: The data type to store the field values in
    """

    def __init__(self, nlen, ilen, n0, i0, current, dtype=np.complex64):
        # -------------
        # INITIAL SETUP
        # -------------
        # Save arguments
        self._nlen = nlen
        self._ilen = ilen
        self._n0 = n0
        self._i0 = i0
        self._dtype = dtype
        # Get and save material dimension info
        if len(np.shape(current)) > 1:
            self._cnlen = np.shape(current)[0]
            self._cilen = np.shape(current)[1]
        else:
            self._cnlen = np.shape(current)[0]
            self._cilen = 1
        # Check for error
        if self._n0 < 0 or self._n0 + self._cnlen > self._nlen:
            raise ValueError("Current cannot start at n=" + str(self._n0) + " and end at n=" + str(self._n0 + self._cnlen) + " as this exceeds the dimensions of the simulation.")
        elif self._i0 < 0 or self._i0 + self._cilen > self._ilen:
            raise ValueError("Current cannot start at i=" + str(self._i0) + " and end at i=" + str(self._i0 + self._cilen) + " as this exceeds the dimensions of the simulation.")
        # Reshape the current array so that it can be indexed correctly
        self._current = np.reshape(current, (self._cnlen, self._cilen))

    def _get_current(self, n):
        """
        Returns the current at time index :math:`n` as an array the length of the simulation
        """
        # Determine if n is within the bounds of the current array
        if n < self._n0 or (self._n0 + self._cnlen) <= n:
            # Not in bounds, return zero-valued array
            return np.zeros(self._cilen, dtype=self._dtype)
        # Pad the current array so that it spans the length of the simulation, note index n-self._n0 is accessed instead of just n
        current_padded = np.pad(self._current[n-self._n0], (self._i0, self._ilen - (self._i0 + self._cilen)), 'constant')
        # Return
        return current_padded


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
    :param storelocs: A list of locations to save chi at during each step in time, indexed from 0 to the material length
    :param dtype: The data type to store the field values in
    """

    def __init__(self, dn, ilen, nlen, mat0, epsiloninf, mata1, mata2, matg, matb, storelocs=[], dtype=np.complex64):
        # -------------
        # INITIAL SETUP
        # -------------
        # Check for error
        if np.shape(mata1) != np.shape(mata2) or np.shape(mata1) != np.shape(matg) or np.shape(mata1) != np.shape(matb):
            raise ValueError("The dimensions of mata1, mata2, matg, and matb should be the same")
        # Save arguments
        self._dn = dn
        self._ilen = ilen
        self._nlen = nlen
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
        # -------------------
        # STORED VALUES SETUP
        # -------------------
        # Save storeloc info
        self._storelocs = storelocs
        self._nlocs = len(self._storelocs)
        # Check to see if any storelocs are requested
        if self._nlocs != 0:
            # Create arrays to store the field values in each location
            self._locs = np.zeros((self._nlen, self._nlocs), dtype=self._dtype)
        # ---------------------
        # CHI CALCULATION SETUP
        # ---------------------
        # Save the chi0 values from chi0_1 and chi0_2 from the indicies we wish to store at all j values
        self._prev_chi_1 = chi0_1[:,self._storelocs]
        self._prev_chi_2 = chi0_2[:,self._storelocs]

    def get_pos(self):
        r"""
        Returns a Numpy array of value 0 outside of the material and 1 inside of the material.

        :return: A Numpy array of value 0 outside of the material and 1 inside of the material
        """
        return np.pad(np.repeat(1, self._matlen), (self._mat0, self._ilen - (self._mat0 + self._matlen)), 'constant')

    def _update_chi(self):
        r"""
        Updates chi_1 and chi_2 (i.e. :math:`\chi_1` and :math:`\chi_2`) using the update equations :math:`\chi^{m+1}_{1,j}=\chi^m_{1,j}e^{\Delta t\left(-\gamma_j+\beta_j\right)}` and :math:`\chi^{m+1}_{2,j}=\chi^m_{2,j}e^{\Delta t\left(-\gamma_j-\beta_j\right)}`. Should be called once per simulation step.
        """
        # Extract the exponents at the j-values we are interested in updating
        exp_1 = self._exp_1[:,self._storelocs]
        exp_2 = self._exp_2[:,self._storelocs]
        # Calculate the updated chi_1 and chi_2
        update_chi_1 = np.multiply(self._prev_chi_1, exp_1)
        update_chi_2 = np.multiply(self._prev_chi_2, exp_2)
        # Save the update_chi_1 and update_chi_2 into the prev_chi_1 and prev_chi_2 values
        self._prev_chi_1 = update_chi_1
        self._prev_chi_2 = update_chi_2

    def _compute_chi(self):
        r"""
        Computes chi at the points specified in the simulation by the `storelocs` parameter via :math:`\chi^n=\Re\left[\chi^n_1e^{\Delta t(-\gamma_j+\beta_j)} + \chi^n_2e^{\Delta t(-\gamma_j-\beta_j)}\right]`.

        :return: :math:`\chi^n` where :math:`n` is the :math:`n` th call to the function `_update_chi`
        """
        # Extract the exponents at the j-values we wish to store
        exp_1 = self._exp_1[:,self._storelocs]
        exp_2 = self._exp_2[:,self._storelocs]
        # Compute chi_1 and chi_2
        t1 = np.multiply(self._prev_chi_1, exp_2)
        t2 = np.multiply(self._prev_chi_2, exp_2)
        # Add chi_1 and chi_2 to yield chi for each oscillator
        chi_j = np.add(t1, t2)
        # Sum across all oscillators to determine chi at each location specified by storelocs
        chi = np.sum(chi_j, axis=0)
        # Return
        return chi

    def _get_epsiloninf(self):
        r"""
        Returns the high frequency susceptability of the material :math:`\epsilon_\infty`.

        :return: A Numpy array of length :code:`ilen` of value 1 outside of the material and :math:`\epsilon_\infty` inside of the material
        """
        return self._epsiloninf

    def _get_chi0(self):
        r"""
        Returns the initial susceptibility of the material :math:`\chi_0`.

        :return: A Numpy array of length :code:`ilen` of value 0 outside of the material and :math:`\chi_0` inside of the material
        """
        return self._chi0

    def _compute_psi(self):
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

    def _update_psi(self, efield):
        """
        Updates the value of psi_1 and psi_2. Should be called once per simulation step.

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

    def _update_mat(self, n, efield):
        """
        Updates the value of psi_1, psi_2, chi_1, and chi_2. Saves the values of chi requested via the `storelocs` parameter.

        :param n: The iteration index :math:`n`
        :param efield: The efield to use in update calculations
        """
        # Update psi and chi
        self._update_psi(efield)
        self._update_chi()
        # Save specific field locations if storing has been requested
        if self._nlocs != 0:
            # Store each location
            self._locs[n,:] = self._compute_chi()

    def export_locs(self):
        """
        Exports the value of chi at a specific location(s) (specified with :code:`storelocs` at initialization) at each point in time.

        :return: A tuple :code:`(ls, locs)` where :code:`ls` is the list :code:`storelocs` (the same :code:`storelocs` that is passed to the Sim class during instantiation), :code:`locs` is a Numpy array containing chi at storelocs at each point in time (axis=0 is time and axis=1 is the respective storeloc location)
        """
        # Check to see if the value of each field at a specific location was saved over time
        if self._nlocs == 0:
            self._locs = None
        # Return
        return (self._storelocs, self._locs)
