.. rcfdtdp documentation master file, created by
   sphinx-quickstart on Fri Jun 29 15:51:12 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rcfdtdpy's documentation!
====================================

RC-FDTD simulations are a staple of electromagnetic field simulations, and can be found in many fields and applications.
This package provides a framework for performing RC-FDTD simulations aimed at investigating a particular problem: the
simulation of materials that have rapidly evolving electric susceptibilities.

The scope of this problem is such that a few assumptions have been made that simplify the implementation of the
simulation as well as the computational complexity of the simulation. These are as follows

- All materials are linear dielectrics such that :math:`P(z,\omega)=\epsilon_0\chi E(z,\omega)`.
- The electric and magnetic fields are plane waves propagating along spatial coordinate :math:`z`.
- Materials are uniform along spatial coordinates :math:`x` and :math:`y`.
- The electric and magnetic fields are zero for all time prior to the start of the simulation (:math:`E(z,t)=0` for all :math:`t<0`).
- The electric field :math:`E(z,t)` is approximately constant over all time intervals of duration :math:`\Delta t`.
- The magnetization of all materials is zero (:math:`\vec{M}=\vec{0}`).

Refer to the RC-FDTD Simulations page in order to learn more about how RC-FDTD simulations work and how simulation
parameters might be tweaked to produce more accurate results. For a description of various simulations run with this
package refer to :download:`this report<files/report.pdf>`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   start
   rcfdtd
   reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
