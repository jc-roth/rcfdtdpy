sim
===

The simulation class treats field indicies as follows

.. image:: ../images/field_structure.png
   :align: center
   :scale: 50

Both the electric and magnetic fields are initialized to the same length, meaning that they have a slight offset in starting and ending locations in the simulation. The E-field at each time is stored in a Numpy array with the following structure

.. image:: ../images/numpy_structure.png
   :align: center
   :scale: 50

The H-field and current arrays have corresponding structures.

* Discuss the choice of :math:`\frac{\Delta t}{\Delta z}\approx c`
* Discuss how to create a current object

.. automodule:: sim
   :members: