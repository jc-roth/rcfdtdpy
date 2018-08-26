from abc import ABC, abstractmethod
import numpy as np
from scipy import integrate
from tqdm import tqdm

"""
Contains the classes used to represent a simulation
"""


class Simulation:
    r"""Represents a single simulation. Field is initialized to all zeros.

    :param i0: The spatial value at which the field starts
    :param i1: The spatial value at which the field ends
    :param di: The spatial step size
    :param n0: The temporal value at which the field starts
    :param n1: The temporal value at which the field ends
    :param dn: The temporal step size
    :param epsilon0: :math:`\epsilon_0`, the vacuum permittivity
    :param mu0: :math:`\mu_0`, the vacuum permeability
    :param boundary: The boundary type of the field, either 'zero', for fields bounded by zeros or 'absorbing' for absorbing boundary conditions
    :param currents: A Current object or a list of Current objects that represent the currents present in the simulation, defaults to none
    :param materials: A Material object or a list of Material objects that represent the materials present in the simulation, defaults to none
    :param nstore: A list of time indices to save field values at in all points in space
    :param istore: A list of spatial indices to save field values at in all points in time
    """

    def __init__(self, i0, i1, di, n0, n1, dn, epsilon0, mu0, boundary, currents=[], materials=[], nstore=[],
                 istore=[]):
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
        elif (not (isinstance(materials, Material) or (type(materials) is list))) or (
                (type(materials) is list) and (len(materials) != 0) and (not isinstance(materials[0], Material))):
            raise TypeError("materials must be either a Material object or a list of Material objects")
        elif (not (isinstance(currents, Current) or (type(currents) is list))) or (
                (type(currents) is list) and (len(currents) != 0) and (not isinstance(currents[0], Current))):
            raise TypeError("currents must be either a Current object or a list of Current objects")
        # Determine the number of temporal and spatial cells in the field
        self._ilen, self._nlen = Simulation.calc_dims(i0, i1, di, n0, n1, dn)
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
        # Put the currents variable into a list if it isn't already
        if isinstance(currents, Current):
            currents = (currents,)
        # List already exits, create an empty current object
        elif len(currents) == 0:
            # Create an empty currents list
            c = np.zeros((1, 1))
            currents = [Current(self._nlen, self._ilen, 0, 0, c)]
        # Save the currents
        self._currents = currents
        # --------------
        # MATERIAL SETUP
        # --------------
        # Put the mat variable into a list if it isn't already
        if isinstance(materials, Material):
            materials = (materials,)
        # List already exits, create an non-interacting material
        elif len(materials) == 0:
            # Create an empty material
            materials = [EmptyMaterial(self._di, self._dn, self._ilen, self._nlen)]
        # Save the material
        self._materials = materials
        # --------------
        # BOUNDARY SETUP
        # --------------
        # Setup boundary condition
        self._boundary = boundary
        if self._boundary == 'absorbing':
            self._eprev0 = np.complex64(0)
            self._eprev1 = np.complex64(0)
            self._erprev0 = np.complex64(0)
            self._erprev1 = np.complex64(0)
        # -----------
        # FIELD SETUP
        # -----------
        # Create each field
        self._efield = np.zeros(self._ilen, dtype=np.complex64)
        self._hfield = np.zeros(self._ilen, dtype=np.complex64)
        # Create each reference field
        self._efieldr = np.zeros(self._ilen, dtype=np.complex64)
        self._hfieldr = np.zeros(self._ilen, dtype=np.complex64)
        # ---------------
        # CONSTANTS SETUP
        # ---------------
        # Save constants
        self._epsilon0 = epsilon0
        self._mu0 = mu0
        # -------------------
        # STORED VALUES SETUP
        # -------------------
        # Save nstore info
        self._nstore = list(nstore)
        self._nstore_len = len(self._nstore)
        # Check to see if any time index stores are requested
        if self._nstore_len != 0:
            # Create arrays to store the field values in each location
            self._nstore_efield = np.zeros((self._nstore_len, self._ilen), dtype=np.complex64)
            self._nstore_hfield = np.zeros((self._nstore_len, self._ilen), dtype=np.complex64)
            self._nstore_efieldr = np.zeros((self._nstore_len, self._ilen), dtype=np.complex64)
            self._nstore_hfieldr = np.zeros((self._nstore_len, self._ilen), dtype=np.complex64)
        # Save istore info
        self._istore = list(istore)
        self._istore_len = len(self._istore)
        # Check to see if any location index stores are requested
        if self._istore_len != 0:
            # Create arrays to store the field values in each location
            self._istore_efield = np.zeros((self._nlen, self._istore_len), dtype=np.complex64)
            self._istore_hfield = np.zeros((self._nlen, self._istore_len), dtype=np.complex64)
            self._istore_efieldr = np.zeros((self._nlen, self._istore_len), dtype=np.complex64)
            self._istore_hfieldr = np.zeros((self._nlen, self._istore_len), dtype=np.complex64)

    def simulate(self, tqdmarg={}):
        """
        Executes the simulation.

        :param tqdmarg: The arguments to pass the tdqm iterator (lookup arguments on the tqdm documentation)
        """
        # Iterate through all materials and reset each so that prior simulation information isn't held in the material.
        for mat in self._materials:
            mat.reset_material()
        # Create a counter for the nstore index
        nstore_index = 0
        # Simulate by iterating over the simulation length
        for n in tqdm(range(self._nlen), **tqdmarg):
            # Update materials
            self._update_materials(n)
            # Update coefficients
            self._update_coefficients()
            # Compute H-field and update
            self._update_hfield()
            self._update_hfieldr()
            # Compute E-field and update
            self._update_efield(n)
            self._update_efieldr(n)
            # Apply boundary conditions
            if self._boundary == 'zero':  # Zero boundary condition
                pass  # No necessary action
            if self._boundary == 'absorbing':  # Absorbing boundary condition
                # Set the field values at the boundary to the previous value one away from the boundary, this somehow
                # results in absorption, I'm not really sure how... I think it has something to do with preventing any
                # wave reflection, meaning that the field values just end up going to zero. It would be a good idea to
                # ask Ben about this.
                self._efield[0] = self._eprev0
                self._efield[-1] = self._eprev1
                self._efieldr[0] = self._erprev0
                self._efieldr[-1] = self._erprev1
                # Save the field values one away from each boundary for use next iteration
                self._eprev0 = self._efield[1]
                self._eprev1 = self._efield[-2]
                self._erprev0 = self._efieldr[1]
                self._erprev1 = self._efieldr[-2]
            # Save the the fields if at the correct index
            if self._nstore.count(n) == 1:
                self._nstore_hfield[nstore_index] = self._hfield
                self._nstore_efield[nstore_index] = self._efield
                self._nstore_hfieldr[nstore_index] = self._hfieldr
                self._nstore_efieldr[nstore_index] = self._efieldr
                nstore_index += 1
            # Save specific field locations if storing has been requested
            if self._istore_len != 0:
                # Store each location
                self._istore_hfield[n] = self._hfield[self._istore]
                self._istore_efield[n] = self._efield[self._istore]
                self._istore_hfieldr[n] = self._hfieldr[self._istore]
                self._istore_efieldr[n] = self._efieldr[self._istore]

    def _update_hfield(self):
        """
        Updates the H-field to the values at the next iteration. Should be called once per simulation step.
        """
        h_t1 = self._hfield[:-1]
        h_t2 = self._coeff_h1 * (self._efield[1:] - self._efield[:-1])
        self._hfield[:-1] = h_t1 - h_t2

    def _update_efield(self, n):
        """
        Updates the E-field to the values at the next iteration. Should be called once per simulation step.

        :param n: The current temporal index of the simulation.
        """
        e_t1 = self._coeff_e0[1:] * self._efield[1:]
        e_t2 = self._coeff_e1[1:] * self._compute_psi()[1:]
        e_t3 = self._coeff_e2[1:] * (self._hfield[1:] - self._hfield[:-1])
        e_t4 = self._coeff_e3[1:] * self._get_current(n)[1:]
        self._efield[1:] = e_t1 + e_t2 - e_t3 - e_t4

    def _update_hfieldr(self):
        """
        Updates the reference H-field to the values at the next iteration. Should be called once per simulation step.
        """
        h_t1 = self._hfieldr[:-1]
        h_t2 = self._coeff_h1r * (self._efieldr[1:] - self._efieldr[:-1])
        self._hfieldr[:-1] = h_t1 - h_t2

    def _update_efieldr(self, n):
        """
        Updates the reference E-field to the values at the next iteration. Should be called once per simulation step.
        
        :param n: The current temporal index of the simulation.
        """
        e_t1 = self._coeff_e0r * self._efieldr[1:]
        e_t3 = self._coeff_e2r * (self._hfieldr[1:] - self._hfieldr[:-1])
        e_t4 = self._coeff_e3r * self._get_current(n)[1:]
        self._efieldr[1:] = e_t1 - e_t3 - e_t4

    def _update_coefficients(self):
        """
        Computes the coefficients for each update term based on the current Material values.
        """
        # Sum the epsiloninf values of each material at the current time to get the final epsiloninf array
        epsiloninf = np.zeros(self._ilen, dtype=np.complex64)
        for mat in self._materials:
            epsiloninf = np.add(epsiloninf, mat.get_epsiloninf())
        # Sum the chi0 values of each material to get the final epsiloninf array
        chi0 = np.zeros(self._ilen, dtype=np.complex64)
        for mat in self._materials:
            chi0 = np.add(chi0, mat.get_chi0())
        # Calculate simulation proportionality constants
        self._coeff_e0 = epsiloninf / (epsiloninf + chi0)
        self._coeff_e1 = 1.0 / (epsiloninf + chi0)
        self._coeff_e2 = self._dn / (self._epsilon0 * self._di * (epsiloninf + chi0))
        self._coeff_e3 = self._dn / (self._epsilon0 * (epsiloninf + chi0))
        self._coeff_h1 = self._dn / (self._mu0 * self._di)
        # Create simulation reference proportionality constants (the reference sees chi0=0 and epsiloninf=1)
        self._coeff_e0r = np.complex64(1)
        self._coeff_e2r = self._dn / (self._epsilon0 * self._di)
        self._coeff_e3r = self._dn / self._epsilon0
        self._coeff_h1r = self._dn / (self._mu0 * self._di)

    def _get_current(self, n):
        """
        Calculates the current at all points in the simulation using all the currents in the simulation

        :param n: The temporal index :math:`n` to calculate the current at
        """
        # Create an array to hold the current
        current = np.zeros(self._ilen)
        for c in self._currents:
            current = np.add(current, c.get_current(n))
        # Return
        return current

    def _compute_psi(self):
        """
        Calculates psi at all points in the simulation using all materials in the simulation.
        """
        # Create an array to hold psi
        psi = np.zeros(self._ilen)
        for mat in self._materials:
            psi = np.add(psi, mat.get_psi())
        # Return
        return psi

    def _update_materials(self, n):
        """
        Updates the each material in the simulation using the `_update_mat` function. Should be called once per
        simulation step.
        """
        # Iterate through all materials and update each
        for mat in self._materials:
            mat.update_material(n, self._efield)

    def get_dims(self):
        """
        Returns the dimensions of the simulation.

        :returns: A tuple :code:`(ilen, nlen)` containing the spatial and temporal dimensions in cells
        """
        return self._ilen, self._nlen

    def get_materials(self):
        """
        Returns the list of Material objects present in the simulation.

        :returns: A list of Material objects
        """
        return self._materials

    def export_nfields(self):
        """
        Exports the field values at temporal indices specified by nstore at the Simulation object's initialization,
        where an index along axis=0 corresponds to the corresponding temporal index in nstore. Values along axis=1
        correspond to the spatial index.

        :return: `(hfield, efield, hfieldr, efieldr)` where the suffix `r` corresponds to a reference field. If nstore is unspecified returns None
        """
        if self._nstore_len == 0:
            return None
        else:
            return self._nstore_hfield, self._nstore_efield, self._nstore_hfieldr, self._nstore_efieldr

    def export_ifields(self):
        """
        Exports the field values at spatial indices specified by istore at the Simulation object's initialization,
        where an index along axis=1 corresponds to the corresponding spatial index in istore. Values along axis=0
        correspond to the temporal index.

        :return: `(hfield, efield, hfieldr, efieldr)` where the suffix `r` corresponds to a reference field. If istore is unspecified returns None
        """
        if self._istore_len == 0:
            return None
        else:
            return self._istore_hfield, self._istore_efield, self._istore_hfieldr, self._istore_efieldr

    @staticmethod
    def calc_dims(i0, i1, di, n0, n1, dn):
        """
        Calculates the dimensions of the simulation in cells.

        :param i0: The spatial value at which the field starts
        :param i1: The spatial value at which the field ends
        :param di: The spatial step size
        :param n0: The temporal value at which the field starts
        :param n1: The temporal value at which the field ends
        :param dn: The temporal step size
        :return: A tuple `(ilen, nlen)` of the spatial and temporal dimensions
        """
        nlen = int(np.floor((n1 - n0) / dn))
        ilen = int(np.floor((i1 - i0) / di) + 2)  # Add two to account for boundary conditions
        return ilen, nlen

    @staticmethod
    def calc_arrays(i0, i1, di, n0, n1, dn):
        """
        Calculates spatial and time arrays of the same dimensions of the simulation. Array values are populated by their
        the spatial and temporal values at their respective simulation spatial and temporal indices.

        :param i0: The spatial value at which the field starts
        :param i1: The spatial value at which the field ends
        :param di: The spatial step size
        :param n0: The temporal value at which the field starts
        :param n1: The temporal value at which the field ends
        :param dn: The temporal step size
        :return: A tuple `(z, t)` of the spatial and temporal arrays
        """
        # Calculate simulation dimensions
        ilen, nlen = Simulation.calc_dims(i0, i1, di, n0, n1, dn)
        # Create z and t arrays
        z = np.linspace(i0 + di / 2, i1 + di / 2, ilen, endpoint=False)
        t = np.linspace(n0 + dn / 2, n1 + dn / 2, nlen, endpoint=False)
        # Return
        return z, t


class Current:
    r"""
    The Current class is used to represent a current in the simulation.

    :param i0: The starting spatial index of the current
    :param n0: The starting temporal index of the current
    :param ilen: The number of spatial indices in the simulation
    :param nlen: The number of temporal indices in the simulation
    :param current: A matrix representing the current, where axis=0 represents locations in time :math:`n` and axis=1 represents locations in space :math:`i`
    """

    def __init__(self, i0, n0, ilen, nlen, current):
        # -------------
        # INITIAL SETUP
        # -------------
        # Save arguments
        self._nlen = nlen
        self._ilen = ilen
        self._n0 = n0
        self._i0 = i0
        # Get and save material dimension info
        if len(np.shape(current)) > 1:
            self._cnlen = np.shape(current)[0]
            self._cilen = np.shape(current)[1]
        else:
            self._cnlen = np.shape(current)[0]
            self._cilen = 1
        # Check for error
        if self._n0 < 0 or self._n0 + self._cnlen > self._nlen:
            raise ValueError("Current cannot start at n=" + str(self._n0) + " and end at n=" + str(
                self._n0 + self._cnlen) + " as this exceeds the dimensions of the simulation.")
        elif self._i0 < 0 or self._i0 + self._cilen > self._ilen:
            raise ValueError("Current cannot start at i=" + str(self._i0) + " and end at i=" + str(
                self._i0 + self._cilen) + " as this exceeds the dimensions of the simulation.")
        # Reshape the current array so that it can be indexed correctly
        self._current = np.reshape(current, (self._cnlen, self._cilen))

    def get_current(self, n):
        """
        Returns the current at time index :math:`n` as an array the length of the simulation
        """
        # Determine if n is within the bounds of the current array
        if n < self._n0 or (self._n0 + self._cnlen) <= n:
            # Not in bounds, return zero-valued array
            return np.zeros(self._cilen, dtype=np.complex64)
        # Pad the current array so that it spans the length of the simulation
        current_padded = np.pad(self._current[n - self._n0], (self._i0, self._ilen - (self._i0 + self._cilen)),
                                'constant')
        # Return
        return current_padded


class Material(ABC):
    r"""
    The Material class is an abstract class that defines the minimum requirements for a Material object to have in the
    simulation. All Materials in the simulation must inherit Material.

    :param di: The spatial time step of the simulation
    :param dn: The temporal step size of the simulation
    :param ilen: The number of spatial indices in the simulation
    :param nlen: The number of temporal indices in the simulation
    :param material_i0: The starting spatial index of the material
    :param material_i1: The ending spatial index of the material
    :param material_n0: The starting temporal index of the material
    :param material_n1: The ending temporal index of the material
    """

    def __init__(self, di, dn, ilen, nlen, material_i0, material_i1, material_n0, material_n1):
        # Call super
        super().__init__()
        # -------------
        # INITIAL SETUP
        # -------------
        # Check for errors
        if material_i1 <= material_i0:
            raise ValueError(
                'The material spatial ending index must be greater than the material spatial starting index')
        elif material_n1 <= material_n0:
            raise ValueError(
                'The material temporal ending index must be greater than the material temporal starting index')
        elif material_i1 - material_i0 > ilen:
            raise ValueError('The material spatial length cannot be greater than the simulation spatial length')
        elif material_n1 - material_n0 > nlen:
            raise ValueError('The material temporal length cannot be greater than the simulation temporal length')
        # Save function arguments
        self._di = di
        self._dn = dn
        self._ilen = ilen
        self._nlen = nlen
        self._material_i0 = material_i0
        self._material_n0 = material_n0
        self._material_ilen = material_i1 - material_i0
        self._material_nlen = material_n1 - material_n0

    @abstractmethod
    def reset_material(self):
        """
        This function is called before each simulation. It should reset any material values that are calculated during
        the simulation to their initial values.
        """
        pass

    @abstractmethod
    def update_material(self, n, efield):
        """
        This function is called at the start of each simulation time step. It should update the :math:`\chi` and
        :math:`\psi` values of the material to their values at n.

        :param n: The current temporal index of the simulation.
        :param efield: The previous electric field of the simulation.
        """
        pass

    @abstractmethod
    def get_chi0(self):
        """
        Returns the value of :math:`\chi_0` at the current time step in the simulation at each spatial location in the
        simulation.

        :return: The current value of :math:`\chi_0` at each spatial location in the simulation
        """
        pass

    @abstractmethod
    def get_epsiloninf(self):
        """
        Returns the value of :math:`\epsilon_\infty` at the current time step in the simulation at each spatial location
        in the simulation.

        :return: The current value of :math:`\epsilon_\infty` at each spatial location in the simulation
        """
        pass

    @abstractmethod
    def get_psi(self):
        """
        Returns the value of :math:`\psi` at the current time step in the simulation at each spatial location in the
        simulation.

        :return: The current value of :math:`\psi` at each spatial location in the simulation
        """
        pass


class EmptyMaterial(Material):
    """
    Represents an empty Material, i.e. vacuum
    """

    def __init__(self, di, dn, ilen, nlen):
        # Call super
        super().__init__(di, dn, ilen, nlen, 0, 1, 0, 1)

    def update_material(self, n, efield):
        # Do nothing
        pass

    def get_chi0(self):
        # Simply return 0
        return np.zeros(self._ilen, dtype=np.complex64)

    def get_epsiloninf(self):
        # Simply return 1
        return np.ones(self._ilen, dtype=np.complex64)

    def get_psi(self):
        # Simply return 0
        return np.zeros(self._ilen, dtype=np.complex64)

    def reset_material(self):
        # Do nothing
        pass


class StaticMaterial(Material):
    """
    The StaticMaterial class allows for the simulation of a static material, that is a material that has a constant
    definition of electric susceptibility in time. The electric susceptibility is modeled using a harmonic oscillator.
    This material is more computationally efficient than NumericMaterial, and uses the update equations specified in
    Beard et al..

    :param di: The spatial time step of the simulation
    :param dn: The temporal step size of the simulation
    :param ilen: The number of spatial indices in the simulation
    :param nlen: The number of temporal indices in the simulation
    :param material_i0: The starting spatial index of the material
    :param epsiloninf: The :math:`\epsilon_\infty` of the material, which is constant over space and time.
    :param a1: A matrix representing :math:`A_1` where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial index
    :param a2: A matrix representing :math:`A_2` where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial index
    :param g: A matrix representing :math:`\gamma` where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial index
    :param b: A matrix representing :math:`\beta` where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial index
    :param opacity: A vector representing the opacity of the material in time. Each index corresponds to the :math:`n` th time index of the material where `1` corresponds to the material being opaque and `0` corresponds to the material being transparent. Values can be real numbers. Defaults to an opaque material for all time.
    :param istore: A list of spatial indices to save :math:`\chi` values at in all points in time
    """

    def __init__(self, di, dn, ilen, nlen, material_i0, epsiloninf, a1, a2, g, b, opacity=None, istore=[]):
        # -------------
        # INITIAL SETUP
        # -------------
        # Check for error
        if np.shape(a1) != np.shape(a2) or np.shape(a1) != np.shape(g) or np.shape(a1) != np.shape(b):
            raise ValueError("The dimensions of a1, a2, g, and b should be the same")
        elif (opacity is not None) and (len(np.shape(opacity)) != 1):
            raise ValueError("opacity should be a 1-dimensional Numpy array of length nlen or None type")
        elif (opacity is not None) and (np.shape(opacity)[0] != nlen):
            raise ValueError("opacity should be a Numpy array of length nlen or None type")
        # Get and save material dimension info
        if len(np.shape(a1)) > 1:
            self._jlen = np.shape(a1)[0]
            material_i1 = material_i0 + np.shape(a1)[1]
        else:
            self._jlen = 1
            material_i1 = material_i0 + np.shape(a1)[0]
        # Call super
        super().__init__(di, dn, ilen, nlen, material_i0, material_i1, 0, nlen)
        # If opacity is unspecified, set equal to opaque for all time, else save provided opacity
        if opacity is None:
            self._opacity = np.ones(self._nlen)
        else:
            self._opacity = opacity
        # Reshape arrays so that they can be indexed correctly
        self._a1 = np.reshape(a1, (self._jlen, self._material_ilen))
        self._a2 = np.reshape(a2, (self._jlen, self._material_ilen))
        self._g = np.reshape(g, (self._jlen, self._material_ilen))
        self._b = np.reshape(b, (self._jlen, self._material_ilen))
        # Epsilon_infinity is equal to one in vacuum, so only set self._epsiloninf equal to epsiloninf in the material
        epsiloninf_repeat = np.repeat(epsiloninf, self._material_ilen)
        self._epsiloninf = np.pad(epsiloninf_repeat,
                                  (self._material_i0, self._ilen - (self._material_i0 + self._material_ilen)),
                                  'constant', constant_values=1)
        # --------------
        # MATERIAL SETUP
        # --------------
        # Calculate susceptibility beta and gamma sums and exponents
        b_min_g = np.add(self._b, -self._g)
        min_b_min_g = np.add(-self._b, -self._g)
        self._exp_1 = np.exp(np.multiply(b_min_g, self._dn))
        self._exp_2 = np.exp(np.multiply(min_b_min_g, self._dn))
        # Calculate initial susceptibility values
        self._chi0_1 = np.zeros((self._jlen, self._material_ilen), dtype=np.complex64)
        self._chi0_2 = np.zeros((self._jlen, self._material_ilen), dtype=np.complex64)
        for j in range(self._jlen):
            for mi in range(self._material_ilen):
                if np.abs(b_min_g[j, mi]) < 1e-5:
                    # beta-gamma is small, avoid divide by zero error
                    self._chi0_1[j, mi] = self._a1[j, mi] * self._dn
                    self._chi0_2[j, mi] = np.multiply(np.divide(self._a2[j, mi], min_b_min_g[j, mi]),
                                                      np.subtract(self._exp_2[j, mi], 1))
                else:
                    # beta-gamma is not small, calculate normally
                    self._chi0_1[j, mi] = np.multiply(np.divide(self._a1[j, mi], b_min_g[j, mi]),
                                                      np.subtract(self._exp_1[j, mi], 1))
                    self._chi0_2[j, mi] = np.multiply(np.divide(self._a2[j, mi], min_b_min_g[j, mi]),
                                                      np.subtract(self._exp_2[j, mi], 1))
        # Calculate first delta susceptibility values
        self._dchi0_1 = np.multiply(self._chi0_1, np.subtract(1, self._exp_1))
        self._dchi0_2 = np.multiply(self._chi0_2, np.subtract(1, self._exp_2))
        # Initialize psi values to zero
        self._psi_1 = np.zeros((self._jlen, self._material_ilen), dtype=np.complex64)
        self._psi_2 = np.zeros((self._jlen, self._material_ilen), dtype=np.complex64)
        # Calculate chi0
        chi0_j = np.add(self._chi0_1, self._chi0_2)
        chi0_summed = np.sum(chi0_j, axis=0)
        # Pad chi0 so that it spans the length of the simulation
        self._chi0 = np.pad(chi0_summed, (self._material_i0, self._ilen - (self._material_i0 + self._material_ilen)),
                            'constant')
        # Create a place to store susceptibility values
        self._chi_1 = np.copy(self._chi0_1)
        self._chi_2 = np.copy(self._chi0_2)
        # -------------------
        # STORED VALUES SETUP
        # -------------------
        # Save istore info
        self._istore = istore
        self._istore_len = len(self._istore)
        # Check to see if any istores are requested
        if self._istore_len != 0:
            # Create arrays to store the field values in each location
            self._istore_chi = np.zeros((self._nlen, self._istore_len), dtype=np.complex64)

    def reset_material(self):
        # Reset psi
        self._psi_1 = np.zeros((self._jlen, self._material_ilen), dtype=np.complex64)
        self._psi_2 = np.zeros((self._jlen, self._material_ilen), dtype=np.complex64)
        # Reset chi
        self._chi_1 = self._chi0_1
        self._chi_2 = self._chi0_2
        # Reset istore_chi
        if self._istore_len != 0:
            # Create arrays to store the field values in each location
            self._istore_chi = np.zeros((self._nlen, self._istore_len), dtype=np.complex64)

    def update_material(self, n, efield):
        """
        Updates the values of :math:`\psi` and :math:`\chi` Saves the values of :math:`chi` requested via the `istore`
        parameter.

        :param n: The iteration index :math:`n`
        :param efield: The efield to use in update calculations
        """
        # Update psi
        self._update_psi(n, efield)
        # Update chi_1 and chi_2
        self._chi_1 = np.multiply(self._chi_1, self._exp_1)
        self._chi_2 = np.multiply(self._chi_2, self._exp_2)
        # Save specific field locations if storing has been requested
        if self._istore_len != 0:
            # Add chi_1 and chi_2 to yield chi for each oscillator at each location specified by istore
            chi_j = np.add(self._chi_1[:, self._istore], self._chi_2[:, self._istore])
            # Sum across all oscillators to determine chi and store
            self._istore_chi[n] = np.sum(chi_j, axis=0)

    def _update_psi(self, n, efield):
        """
        Updates the value of psi_1, psi_2, and psi_opacity. Should be called once per simulation step.

        :param efield: The efield to use in update calculations
        """
        # Copy the efield so that instead of being a vector it is a matrix composed of horizontal efield vectors
        e = np.tile(efield[self._material_i0:self._material_i0 + self._material_ilen], (self._jlen, 1))
        # Calculate first term
        t1_1 = np.multiply(e, self._dchi0_1)
        t1_2 = np.multiply(e, self._dchi0_2)
        # Calculate second term
        t2_1 = np.multiply(self._psi_1, self._exp_1)
        t2_2 = np.multiply(self._psi_2, self._exp_2)
        # Update next psi values
        self._psi_1 = np.add(t1_1, t2_1)
        self._psi_2 = np.add(t1_2, t2_2)
        # Update psi opacity values
        self._psi_opacity = self._opacity[n]

    def get_chi0(self):
        """
        Returns the value of :math:`\chi^0` at all spatial indices resulting from the material.

        :return: :math:`\chi^0` at all values of the simulation resulting from the material.
        """
        return self._chi0

    def get_epsiloninf(self):
        """
        Returns the value of :math:`\epsilon_\infty` at all spatial indices resulting from the material.

        :return: :math:`\epsilon_\infty` at all values of the simulation resulting from the material.
        """
        return self._epsiloninf

    def get_psi(self):
        """
        Calculates psi at all points in the simulation using the current value of psi_1 and psi_2. Scaled by the
        `opacity` array passed in at initialization.

        :return:  :math:`\psi^n` at all values of the simulation resulting from the material where :math:`n` is given by the number of times `update_material` has been called by the `Material`'s corresponding `Simulation` object, which is once per simulation step.
        """
        # Find the psi matrix
        psi_j = np.add(self._psi_1, self._psi_2)
        # Sum the psi matrix along axis=0 to combine all oscillators
        psi = np.sum(psi_j, axis=0)
        # Pad the psi array so that it spans the length of the simulation
        psi_padded = np.pad(psi, (self._material_i0, self._ilen - (self._material_i0 + self._material_ilen)),
                            'constant')
        # Return the real part as specified in Beard
        return np.real(psi_padded * self._psi_opacity)

    def export_ifields(self):
        """
        Exports the :math:`\chi` values at spatial indices specified by istore at the Material object's initialization,
        where an index along axis=1 corresponds to the corresponding spatial index in istore. Values along axis=0
        correspond to the temporal index.

        :return: `chi` or None if istore is unspecified.
        """
        if self._istore_len == 0:
            return None
        else:
            return self._istore_chi


class NumericMaterial(Material):
    r"""
    The NumericMaterial class represents a material that has a non-constant definition of electric susceptibility in
    time. Currently this Material can only represent materials that have a constant definition of electric
    susceptibility in space.

    :param di: The spatial time step of the simulation
    :param dn: The temporal step size of the simulation
    :param ilen: The number of spatial indices in the simulation
    :param nlen: The number of temporal indices in the simulation
    :param material_i0: The starting spatial index of the material
    :param material_i1: The ending spatial index of the material
    :param chi_func: A function representing the electric susceptibility :math:`\chi` as a function of time. The function should accept a single argument n corresponding to the time index.
    :param epsiloninf: A function representing the :math:`\epsilon_\infty` of the material as a function of time. The function should accept a single argument n corresponding to the time index.
    :param tqdmarg: The arguments to pass the tdqm iterator (lookup arguments on the tqdm documentation).
    """

    def __init__(self, di, dn, ilen, nlen, material_i0, material_i1, chi_func, epsiloninf_func, tqdmarg={}):
        # -------------
        # Initial setup
        # -------------
        # Call super
        super().__init__(di, dn, ilen, nlen, material_i0, material_i1, 0, nlen)
        # Save the infinity permittivity function
        self._epsiloninf_func = epsiloninf_func
        # Set self._epsiloninf to a zero valued array
        self._epsiloninf = np.zeros(self._ilen, dtype=np.complex64)
        # --------------------
        # Setup susceptibility
        # --------------------
        # Wrap chi_func to account for real and imaginary parts
        def chi_func_real(t):
            return np.real(chi_func(t))
        def chi_func_imag(t):
            return np.imag(chi_func(t))
        # Create an array to hold chi values at specific values of m
        self._chi_m = np.zeros(nlen, dtype=np.complex64)
        # Create an array to hold dchi values at specific values of m
        self._dchi_m = np.zeros((nlen - 1, 1), dtype=np.complex64)
        # Calculate chi_m at m=0 and store
        real_area, real_area_err = integrate.quad(chi_func_real, 0, dn)
        imag_area, imag_area_err = integrate.quad(chi_func_imag, 0, dn)
        # Combine real and imaginary parts
        self._chi_m[0] = real_area + 1j*imag_area
        chi0_repeat = np.repeat(self._chi_m[0], self._material_ilen)
        # chi is zero in vacuum, so pad chi0 in the material by zero outside of the material
        self._chi0 = np.pad(chi0_repeat, (self._material_i0, self._ilen - (self._material_i0 + self._material_ilen)),
                            'constant', constant_values=0)
        # Iterate over all m and integrate at each to find dchi_m
        for m in tqdm(range(1, nlen), **tqdmarg):
            # Integrate real and imaginary parts
            real_area, real_area_err = integrate.quad(chi_func_real, m * dn, (m + 1) * dn)
            imag_area, imag_area_err = integrate.quad(chi_func_imag, m * dn, (m + 1) * dn)
            # Combine real and imaginary parts
            area = real_area + 1j*imag_area
            # Store
            self._chi_m[m] = area
            self._dchi_m[m - 1] = self._chi_m[m - 1] - self._chi_m[m]
        # Create a 2D Numpy array to hold previous values of the electric field
        self._efield = np.zeros((nlen, self._material_ilen), dtype=np.complex64)
        # Create a 1D Numpy array to hold the current value of psi
        self._psi = np.zeros(self._material_ilen, dtype=np.complex64)

    def reset_material(self):
        # Clear the electric field
        self._efield = np.zeros((self._nlen, self._material_ilen), dtype=np.complex64)
        # Clear the psi array
        self._psi = np.zeros(self._material_ilen, dtype=np.complex64)

    def update_material(self, n, efield):
        # ------------------------------
        # Electric susceptibility update
        # ------------------------------
        # Save the current efield value
        self._efield[n] = efield[self._material_i0:self._material_i0 + self._material_ilen]
        # Trim dchi_m to length n, transpose, flip, and repeat to the length of the material
        dchi = np.tile(np.flip(self._dchi_m[:n], 0), (1, self._material_ilen))
        # Trim the efield to length n (shifted by one as specified by update equations)
        e = self._efield[1:n+1]
        # Multiply e and dchi and sum to determine psi
        self._psi = np.sum(np.multiply(e, dchi), axis=0, dtype=np.complex64)
        # ----------------------------
        # Infinite permittivity update
        # ----------------------------
        # Calculate epsiloninf at the current time
        epsiloninf = self._epsiloninf_func(self._dn * (n + 0.5))
        # Epsilon_infinity is equal to one in vacuum, so only set self._epsiloninf equal to epsiloninf in the material
        epsiloninf_repeat = np.repeat(epsiloninf, self._material_ilen)
        self._epsiloninf = np.pad(epsiloninf_repeat,
                                  (self._material_i0, self._ilen - (self._material_i0 + self._material_ilen)),
                                  'constant', constant_values=1)

    def get_chi0(self):
        return self._chi0

    def get_epsiloninf(self):
        return self._epsiloninf

    def get_psi(self):
        # Pad the psi array so that it spans the length of the simulation
        psi_padded = np.pad(self._psi, (self._material_i0, self._ilen - (self._material_i0 + self._material_ilen)),
                            'constant')
        # Return the real part as specified in Beard
        return np.real(psi_padded)

    def export_chi(self):
        """
        Exports the electric susceptibility :math:`\chi` at each time index.

        :return: The electric susceptibility :math:`\chi`
        """
        return self._chi_m


class TwoStateMaterial(Material):
    r"""
    The TwoStateMaterial class represents a material that has a non-constant definition of electric susceptibility in
    time. This material is represents the same material proposed in Beard and Schmuttenmaer 2001
    (www.doi.org/10.1063/1.1338526). The ground state electric susceptibility oscillator is defined by

    :math:`\chi_{g,j}(t)=e^{-\gamma_{g,j}t}\left[A_{g,1,j}e^{\beta_{g,j}t}+A_{g,2,j}e^{-\beta_{g,j}t}\right]`

    and the excited state electric susceptibility oscillator is defined by

    :math:`\chi_{e,j}(t)=e^{-\gamma_{e,j}t}\left[A_{e,1,j}e^{\beta_{e,j}t}+A_{e,2,j}e^{-\beta_{e,j}t}\right]`

    There is also an visual electric susceptibility oscillator that represents any susceptibility change resulting
    directly from the incident visual pulse. This is defined by

    :math:`\chi_{\text{vis},j}(t)=e^{-\gamma_{\text{vis},j}t}\left[A_{\text{vis},1,j}e^{\beta_{\text{vis},j}t}+A_{\text{vis},2,j}e^{-\beta_{\text{vis},j}t}\right]`

    The total susceptibility is defined as

    :math:`\chi(t,t'',z)=\sum_j\left[f_e(t,t'',z)\chi_{e,j}(t)+(1-f_e(t,t'',z))\chi_{g,j}(t)+g(t,t'',z)\chi_{\text{vis},j}(t)\right]`

    where :math:`f_e(t,t'',z)` the the fraction of excited oscillators and :math:`g(t,t'',z)` is the incident visual
    pulse. We define :math:`f_e(t,t'',z)` as a decay as a function of distance into the material times the convolution
    of the visual pulse and the excited state decay

    :math:`f_e(t,t'',z)=\exp{\left(-\alpha z\right)}\int_0^t\exp{\left[-\left(\frac{t''-t'}{\Gamma}\right)^2\right]}\left(\exp{\left[-\frac{t'}{\tau}\right]}+b\right)\text{d}t'`

    where :math:`\alpha` is the spatial decay of the visual pulse in the material, :math:`\Gamma` is the width of the
    visual pulse, :math:`t''` is the time delay between the visual and simulation pulses, :math:`\tau` is the time decay
    of an excited oscillator, :math:`b` is the offset that the excited oscillators have (used if not all excited
    oscillators ultimately de-excite), and :math:`t'=t-zv_\text{vis}`. We define :math:`g(t,t'',z)` as the visual pulse
    multiplied by the spatial decay of the pulse in the material

    :math:`g(t,t'',z)=\exp{\left(-\alpha z\right)}\exp{\left[-\left(\frac{t''-t'}{\Gamma}\right)^2\right]}`

    The user supplies :math:`\alpha`, :math:`\Gamma`, :math:`t''`, :math:`\tau`, and :math:`b` at the simulation
    initialization to define :math:`f_e(t,t'',z)` and :math:`g(t,t'',z)`. The user must also supply the constants that
    define the ground, excited, and visual oscillators. Initializing the material will trigger the computation of the
    fraction of excited oscillators followed by the computation of the discretized electric susceptibility.

    :param di: The spatial time step of the simulation
    :param dn: The temporal step size of the simulation
    :param ilen: The number of spatial indices in the simulation
    :param nlen: The number of temporal indices in the simulation
    :param n_ref_ind: The time index to use as a reference for the the `t_diff` argument
    :param material_i0: The starting spatial index of the material
    :param e_a1: The excited state oscillator :math:`A_{e,1}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param e_a2: The excited state oscillator :math:`A_{e,2}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param e_b: The excited state oscillator :math:`\beta_{e,1}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param e_g: The excited state oscillator :math:`\gamma_{e,1}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param g_a1: The ground state oscillator :math:`A_{e,1}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param g_a2: The ground state oscillator :math:`A_{e,2}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param g_b: The ground state oscillator :math:`\beta_{e,1}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param g_g: The ground state oscillator :math:`\gamma_{e,1}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param v_a1: The visual oscillator :math:`A_{e,1}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param v_a2: The visual oscillator :math:`A_{e,2}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param v_b: The visual oscillator :math:`\beta_{e,1}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param v_g: The visual oscillator :math:`\gamma_{e,1}`, where axis=0 represents the :math:`j` th oscillator and axis=1 represents the :math:`i` th spatial oscillator in the material.
    :param alpha: The spatial decay constant :math:`\alpha` in meters
    :param Gamma: The visual pulse width :math:`\Gamma` in seconds
    :param t_diff: The time difference :math:`t''` in seconds between the visual pulse and index `n_ref_ind`
    :param tau: The oscillator time decay constant :math:`\tau` in seconds
    :param b: The excited oscillator decay offset :math:`\b`
    :param epsiloninf: The :math:`\epsilon_\infty` of the material, which is constant over space and time.
    :param tqdmarg_f: The arguments to pass the tdqm iterator during the fraction of excited oscillators calculation (lookup arguments on the tqdm documentation).
    :param tqdmarg_c: The arguments to pass the tdqm iterator during the electric susceptibility calculation (lookup arguments on the tqdm documentation).
    """

    def __init__(self, di, dn, ilen, nlen, n_ref_ind, material_i0, e_a1, e_a2, e_b, e_g, g_a1, g_a2, g_b, g_g, v_a1, v_a2, v_b, v_g, alpha, Gamma, t_diff, tau, b, epsiloninf, tqdmarg_f={}, tqdmarg_c={}):
        # -------------
        # INITIAL SETUP
        # -------------
        # Check for errors
        shape = np.shape(e_a1)
        if shape != np.shape(e_a2) or shape != np.shape(e_b) or shape != np.shape(e_g) or shape != np.shape(g_a1) or shape != np.shape(g_a2) or shape != np.shape(g_g) or shape != np.shape(g_b) or shape != np.shape(v_a1) or shape != np.shape(v_a2) or shape != np.shape(v_g) or shape != np.shape(v_b):
            raise ValueError("e_a1, e_a2, e_b, e_g, g_a1, g_a2, g_b, g_g, v_a1, v_a2, v_b, v_g must all have the same dimensions")
        elif n_ref_ind > nlen:
            raise ValueError("n_ref_ind cannot be larger then nlen, the length of the simulation")
        # Get and save material dimension info
        if len(shape) > 1:
            self._jlen = shape[0]
            material_i1 = material_i0 + shape[1]
        else:
            self._jlen = 1
            material_i1 = material_i0 + shape[0]
        # Call super
        super().__init__(di, dn, ilen, nlen, material_i0, material_i1, 0, nlen)
        # -----------------------------------
        # COMPUTE EXCITED OSCILLATOR FRACTION
        # -----------------------------------

        # Define the integrand (i.e. gaussian multiplied by exponential decay)
        def integrand(tp):
            p1 = np.exp(-np.square(np.divide(t_diff - tp, Gamma)))
            p2 = np.add(np.exp(-np.divide(tp, tau)), b)
            return np.multiply(p1, p2)
        # Create an array to hold the fraction of excited oscillators
        f_e = np.zeros((nlen, 1))
        # Calculate the fraction of excited oscillators at each time
        for i in tqdm(range(nlen), **tqdmarg_f):
            vis_time = (i - n_ref_ind) * dn
            area, err = integrate.quad(integrand, -nlen * dn, vis_time)
            f_e[i] = area
        # Tile f_e along spatial extent of material
        f_e = np.tile(f_e, (1, ilen))
        # Compute spatial decay coefficient for each location
        spatial_locs = np.arange(0, self._material_ilen * di, di)
        spatial_coeffs = np.exp(-np.divide(spatial_locs, tau))
        spatial_coeffs = np.tile(spatial_coeffs, (1, nlen))
        # Multiply each term in f_e by the spatial coefficients to apply the spatial absorption decay
        self._f_e = np.multiply(spatial_coeffs, f_e)
        # -------------------------------------------
        # COMPUTE DISCRETIZED ELECTRIC SUSCEPTIBILITY
        # -------------------------------------------

        # Define the excited state oscillator electric susceptibility function
        def e_chi(t):
            return np.multiply(np.exp(-e_g * t), np.add(e_a1*np.exp(e_b*t), e_a2*np.exp(-e_b*t)))

        # Define the ground state oscillator electric susceptibility function
        def g_chi(t):
            return np.multiply(np.exp(-g_g * t), np.add(g_a1*np.exp(g_b*t), g_a2*np.exp(-g_b*t)))

        # Define the visual oscillator electric susceptibility function
        def v_chi(t):
            return np.multiply(np.exp(-v_g * t), np.add(v_a1*np.exp(v_b*t), v_a2*np.exp(-v_b*t)))

        # Create an array to hold chi values at specific values of m
        chi_m = np.zeros((self._jlen, self._material_ilen, nlen), dtype=np.complex64)
        # Create an array to hold dchi values at specific values of m
        self._dchi_m = np.zeros((self._jlen, self._material_ilen, nlen - 1), dtype=np.complex64)
        # TODO PICKUP FROM HERE WITH THE CALCULATION OF CHI0
        # Calculate chi_m at m=0 and store
        chi_m[0], chi_m0_err = integrate.quad(chi_func, 0, dn)
        chi0_repeat = np.repeat(chi_m[0], self._material_ilen)
        # chi is zero in vacuum, so pad chi0 in the material by zero outside of the material
        self._chi0 = np.pad(chi0_repeat, (self._material_i0, self._ilen - (self._material_i0 + self._material_ilen)),
                            'constant', constant_values=0)
        # Iterate over all m and integrate at each to find dchi_m
        for m in tqdm(range(1, nlen), **tqdmarg_c):
            area, area_err = integrate.quad(chi_func, m * dn, (m + 1) * dn)
            chi_m[m] = area
            self._dchi_m[m - 1] = chi_m[m - 1] - chi_m[m]


    def reset_material(self):
        pass

    def update_material(self, n, efield):
        # Save the current efield value
        self._efield[n] = efield[self._material_i0:self._material_i0 + self._material_ilen]
        # Trim dchi_m to length n, transpose, flip, and repeat to the length of the material
        dchi = np.tile(np.flip(self._dchi_m[:n], 0), (1, self._material_ilen))
        # Trim the efield to length n (shifted by one as specified by update equations)
        e = self._efield[1:n+1]
        # Multiply e and dchi and sum to determine psi
        self._psi = np.sum(np.multiply(e, dchi), axis=0, dtype=np.complex64)

    def get_chi0(self):
        pass

    def get_epsiloninf(self):
        pass

    def get_psi(self):
        pass