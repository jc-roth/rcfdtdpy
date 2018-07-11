RC-FDTD
=======

Todo
----
* Ensure that units are plotting correctly in the vis module
* Add unit label to the pcolormesh levels
* Create a 'builder' class which can be used to easily create current and susceptability matrices/tensors
* Create reference fields like Ben has
* Create an ability to save the field values at a single point in space over all time. This can be used to integrate the field that passes through a point
* Make the H-field and E-field updates in their own methods so that code isn't repeated in the :code:`Sim._absorbing` and :code:`Sim._zero methods`