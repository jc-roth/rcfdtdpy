RC-FDTD
=======

Todo
----
* Ensure that units are plotting correctly in the vis module
* Add unit label to the pcolormesh levels
* Create a 'builder' class which can be used to easily create current and susceptability matrices/tensors
* Create material class, allow multiple materials to be added to simulation
* Split the Sim.export function into multiple functions. Currently its functionality is to broad and makes it unweildy.
* Create a tutorial on the documentation
* Add a tips section
    * One tip: if the field lines look very thick, move back your current pulse in time. It is also helpful for current pulses to be smooth (i.e. no step or delta functions), as these discontinuous functions seem to mess up simulation math