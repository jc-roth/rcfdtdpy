.. RCFDTD.py documentation master file, created by
   sphinx-quickstart on Fri Jun 29 15:51:12 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RCFDTD.py's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   rcfdtd
   sim

Ideas
-----
* Parallelize the calculations in time (i.e. calculate each index simultaneously) to speed up.
* Replace the Numpy arrays with h5py datasets, which can store much larger files locally.
* Visualization - Use matplotlib to export heat map of field intensity in space
* Create a VectorField class that can be used for non-plane wave simulations

The Simulation Class
--------------------
The simulation class treats field indicies as follows

.. image:: ../images/field_structure.png
   :align: center
   :scale: 50

Both the electric and magnetic fields are initialized to the same length, meaning that they have a slight offset in starting and ending locations in the simulation. The field at each time is stored in a Numpy array. The Numpy arrays are stored in a list, which allows each field at previous times to be accessed.

* Discuss the choice of :math:`\frac{\Delta t}{\Delta z}\approx c`

The Field Class
---------------
The field class can store both current field values as well as past field values as as Numpy array. The array has the following structure

.. image:: ../images/numpy_structure.png
   :align: center
   :scale: 50

The user never directly interacts with the entire Numpy array. Rather they set a "current time :math:`n`" which selects a row of the array. Any further operations are performed on that row. The :meth:`sim.Field.export` function simply returns the Numpy field.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
