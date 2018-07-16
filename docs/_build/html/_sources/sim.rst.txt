sim
===

The simulation class treats field indicies as follows

.. image:: images/field_structure.png
   :align: center
   :scale: 50

Both the electric and magnetic fields are initialized to the same length, meaning that they have a slight offset in starting and ending locations in the simulation. The E-field at each time is stored in a Numpy array with the following structure

.. image:: images/numpy_structure.png
   :align: center
   :scale: 50

The H-field and current arrays have corresponding structures. The variables :math:`A_1`, :math:`A_2`, :math:`\gamma`, and :math:`\beta` are represented using the following two dimensional array

.. image:: images/material_numpy_structure.png
   :align: center
   :scale: 50

where :math:`m_0` is the starting index of the material. Increments along the vertical axis represent increments in the oscillator index :math:`j`, and increments along the horizontal axis represent increments in space.

* Discuss the choice of :math:`\frac{\Delta t}{\Delta z}\approx c`

.. automodule:: rcfdtd_sim.sim
   :members: