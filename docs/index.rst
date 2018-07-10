.. RCFDTD.py documentation master file, created by
   sphinx-quickstart on Fri Jun 29 15:51:12 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RCFDTD.py's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   sim

Simulation Details
------------------
Calculates the magnetic field according to :math:`H^{i+1/2,n+1/2}=H^{i+1/2,n-1/2}-\frac{1}{\mu_0}\frac{\Delta t}{\Delta z}\left[E^{i+1,n}-E^{i,n}\right]`. Note that the prior electric field array is located half a time index away at :math:`n-1/2` and the prior magnetic field array is located a whole time index away at :math:`n-1`.

Calcualtes the electric field according to :math:`E^{i,n+1}=\frac{\epsilon_\infty}{\epsilon_\infty+\chi_e^0}E^{i,n}+\frac{1}{\epsilon_\infty+\chi_e^0}\psi^n-\frac{1}{\epsilon_0\left[\epsilon_\infty+\chi_e^0\right]}\frac{\Delta t}{\Delta z}\left[H^{i+1/2,n+1/2}-H^{i-1/2,n+1/2}\right]-\frac{\Delta tI_f}{\epsilon_0\left[\epsilon_\infty+\chi_e^0\right]}`. Note that the prior electric field array is located half a time index away at :math:`n-1` and the prior magnetic field array is located half a time index away at :math:`n-1/2`.

Ideas
-----
* Parallelize the calculations in time (i.e. calculate each index simultaneously) to speed up.
* Replace the Numpy arrays with h5py datasets, which can store much larger files locally.
* Visualization - Use matplotlib to export heat map of field intensity in space
* Create a VectorField class that can be used for non-plane wave simulations

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
