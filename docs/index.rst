.. RCFDTD.py documentation master file, created by
   sphinx-quickstart on Fri Jun 29 15:51:12 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RCFDTD.py's documentation!
=====================================

Code Outline
============
Constants
---------
* epsilon_inf
* epsilon_0
* delta_t
* delta_z

Functions
---------
* diff_chi(A1, A2, gamma, beta, delta_t, m)
* psi(n, I, diff_chi variables)
* electric_field_update(chi_0, epsilon_inf, E_array, H_array, psi variables, epsilon_0, delta_t, delta_z, i, n, current)
* magnetic_field_update(mu_0, E_array, H_array, delta_t, delta_z, i, n)

Data Structures
---------------
* magnetic field array (2-dimensional, space, time)
* electric field array (2-dimensional, space, time)

Loops
-----
* Outer loop - Loops over time

  * Inner loop - Loops over space

    * Updates E-field
    * Updates H-field

Tools
-----
* Visualization - Use matplotlib to export heat map of field intensity in space

Ideas
-----
* Parallelize the calculations in time to speed up.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: rcfdtd
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
