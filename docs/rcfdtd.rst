RC-FDTD Simulations
===================

Recursive convolution finite difference time domain (RC-FDTD) simulations have long been used to numerically solve
Maxwell's equations. This simulation technique discretizes the time domain and evolves the electric and magnetic fields
in time using a set of update equations. Within the simulation, space is discretized into intervals of length
:math:`\Delta z` and time into intervals of length :math:`\Delta t`. A specific point in time and space is accessed via
:math:`z=i\Delta z` and :math:`t=n\Delta t`. The simulation relies on a number of assumptions:

- All materials are linear dielectrics such that :math:`P(z,\omega)=\epsilon_0\chi E(z,\omega)`.
- The electric and magnetic fields are plane waves propagating along spatial coordinate :math:`z`.
- Materials are uniform along spatial coordinates :math:`x` and :math:`y`.
- The electric and magnetic fields are zero for all time prior to the start of the simulation (:math:`E(z,t)=0` for all :math:`t<0`).
- The electric field :math:`E(z,t)` is approximately constant over all time intervals of duration :math:`\Delta t`.
- The magnetization of all materials is zero (:math:`\vec{M}=\vec{0}`).

These assumptions allow the derivation of the discretized displacement field :math:`D^{i,n}`. The displacement field
:math:`\vec{D}(\vec{r},\omega)`, with the requirement that simulated materials are linear dielectrics such that
:math:`P(z,\omega)=\epsilon_0\chi(z,\omega) E(z,\omega)` and the requirement that the field varies over only the spatial
coordinate :math:`z` we find that :math:`D(z,\omega)` is

.. math::

    D(z,\omega)=\epsilon_0\left[1+\chi (z,\omega)\right]E(z,\omega)

The displacement field :math:`D(z,\omega)` can be transformed to the time domain via

.. math::

    D(z,t)=&\mathcal{F}^{-1}\left\{D(z,\omega)\right\} \\
    =&\mathcal{F}^{-1}\left\{\epsilon_0\left[1+\chi (\omega)\right]E(z,\omega)\right\} \\
    =&\mathcal{F}^{-1}\left\{\epsilon_0\mathcal{F}\left\{1+\chi (t)\right\}\mathcal{F}\left\{E(z,t)\right\}\right\}

where :math:`\mathcal{F}\left\{a(t)\right\}` and :math:`\mathcal{F}^{-1}\left\{a(\nu)\right\}` to denote Fourier and
inverse Fourier transforms. Thus via the convolution theorem

.. math::

    D(z,t)&=\mathcal{F}^{-1}\left\{\epsilon_0\mathcal{F}\left\{1+\chi (t)\right\}\mathcal{F}\left\{E(z,t)\right\}\right\} \\
    &=\epsilon_0\left[1+\chi (t)\right]*\left[E(z,t)\right] \\
    &=\epsilon_0\left[\epsilon_\infty E(z,t)+\int_0^t\chi (\tau)E(z,t-\tau) d\tau\right]

where :math:`*` denotes a convolution. It is assumed that :math:`E(z,t)=0` for all :math:`t<0`. We discretize this
result by replacing the :math:`z` and :math:`t` coordinates via :math:`z=i\Delta z` and :math:`t=n\Delta t` where
:math:`i,n\in\mathbb{R}`, yielding

.. math::

    D(i\Delta z,n\Delta t)=&\epsilon_0\epsilon_\infty E(i\Delta z,n\Delta t) \\
    &+\epsilon_0\int_0^{n\Delta t}\chi (\tau)E(i\Delta z,n\Delta -\tau) d\tau

Assuming that :math:`E(i\Delta z,n\Delta -\tau)` is constant over all time intervals of duration :math:`\Delta t` the
integral is replaced with a sum

.. math::

    D^{i,n}=\epsilon_0\epsilon_\infty E^{i,n}+\epsilon_0\sum^{n-1}_{m=0}E^{i,n-m}\chi ^m \label{eq:disp}

where

.. math::

    \chi ^m=\int_{m\Delta t}^{(m+1)\Delta t}\chi (\tau) d\tau

It is *not* assumed :math:`\chi(t)` is constant over any time interval. This result is consistent with the result
derived in Luebbers et al. and Beard et al..


.. math::

    D^{i,n}=\epsilon_0\epsilon_\infty E^{i,n}+\epsilon_0\sum^{n-1}_{m=0}E^{i,n-m}\chi ^m

where

.. math::

    \chi ^m=\int_{m\Delta t}^{(m+1)\Delta t}\chi (\tau) d\tau \label{eq:chi_conv}

This result is significant for the RC-FDTD simulation framework implmented here as it means a material can be simulated
as long as one can define $\chi(t)$ for that material.

We proceed by deriving the update equations for the electric and magnetic fields. With the requirement that
:math:`\vec{M}=\vec{0}` and the requirement that the electric and magnetic fields are uniform in spatial coordinates
:math:`x` and :math:`y`, Faraday's law of induction and Ampere's law with Maxwell's addition reduce to

.. math::

    \frac{\partial E}{\partial z}=-\mu_0\frac{\partial H}{\partial t} \qquad -\frac{\partial H}{\partial z}=I_f+\frac{\partial D}{\partial t}

where :math:`I_f` is along :math:`\hat{z}`. Noting the definition of a derivative we find

.. math::

    \lim_{\Delta z\to0}\frac{E(z+\Delta z,t)-E(z,t)}{\Delta z}=-\mu_0\lim_{\Delta t\to0}\frac{H(z,t+\Delta t)-H(z,t)}{\Delta t} \\
    -\lim_{\Delta z\to0}\frac{H(z+\Delta z,t)-H(z,t)}{\Delta z}=I_f+\lim_{\Delta t\to0}\frac{D(z,t+\Delta t)-D(z,t)}{\Delta t}

From here the discretization process is simple. We simply remove each limit from the equations, define an appropriate
value of :math:`\Delta z` and :math:`\Delta t`, and replace the fields with their discretized forms.

.. math::

    \frac{E^{i+1,n}-E^{i,n}}{\Delta z}=-\mu_0\frac{H^{i,n+1}-H^{i,n}}{\Delta t} \label{eq:faraday} \\
    -\frac{H^{i+1,n}-H^{i,n}}{\Delta z}=I_f+\frac{D^{i,n+1}-D^{i,n}}{\Delta t} \label{eq:ampere}

If :math:`\Delta z` and :math:`\Delta t` aren't small enough such that the derivative is accurate then the RC-FDTD
simulation will break down.

We solve Eq.(\ref{eq:faraday}) for :math:`H^{i,n+1}`, finding the following update equation

.. math::

    H^{i,n+1}=H^{i,n}-\frac{1}{\mu_0}\frac{\Delta t}{\Delta z}\left[E^{i+1,n}-E^{i,n}\right]

In order to solve Eq.(\ref{eq:ampere}) we use the result of Eq.(\ref{eq:disp}) to determine :math:`D^{i,n+1}-D^{i,n}` in
terms of :math:`E^{i+1,n}` and :math:`E^{i,n}`

.. math::

    D^{i,n+1}-D^{i,n}&=\epsilon_0\epsilon_\infty E^{i,n+1}+\epsilon_0\sum^{n}_{m=0}E^{i,n+1-m}\chi ^m-\epsilon_0\epsilon_\infty E^{i,n}-\epsilon_0\sum^{n-1}_{m=0}E^{i,n-m}\chi ^m \\
    &=\epsilon_0\epsilon_\infty\left[E^{i,n+1}-E^{i,n}\right]+\epsilon_0\left[\sum^{n}_{m=0}E^{i,n+1-m}\chi ^m-\sum^{n-1}_{m=0}E^{i,n-m}\chi ^m\right]


Noting that

.. math::

    \sum^{n}_{m=0}E^{i,n+1-m}\chi ^m-\sum^{n-1}_{m=0}E^{i,n-m}\chi ^m&=E^{i,n+1}\chi ^0+\sum^{n}_{m=1}E^{i,n+1-m}\chi ^m-\sum^{n-1}_{m=0}E^{i,n-m}\chi ^m \\
    &=E^{i,n+1}\chi ^0+\sum^{n-1}_{m=0}E^{i,n+1-(m+1)}\chi ^{m+1}-\sum^{n-1}_{m=0}E^{i,n-m}\chi ^m \\
    &=E^{i,n+1}\chi ^0+\sum^{n-1}_{m=0}E^{i,n-m}\left[\chi ^{m+1}-\chi ^m\right] \\

and letting

.. math::

    \Delta\chi^m&=\chi^m-\chi^{m+1} \\
    \psi^n&=\sum^{n-1}_{m=0}E^{i,n-m}\Delta\chi^m

we find

.. math::

    D^{i,n+1}-D^{i,n}&=\epsilon_0\epsilon_\infty\left[E^{i,n+1}-E^{i,n}\right]+\epsilon_0\left[E^{i,n+1}\chi^0-\psi^n\right] \\
    &=\epsilon_0\left[\epsilon_\infty+\chi^0\right]E^{i,n+1}-\epsilon_0\epsilon_\infty E^{i,n}-\epsilon_0\psi^n

Substituting this result into Eq.(\ref{eq:ampere}) and solving for :math:`E^{i,n+1}` we find

.. math::

    E^{i,n+1}=&\frac{\epsilon_\infty}{\epsilon_\infty+\chi^0}E^{i,n}+\frac{1}{\epsilon_\infty+\chi^0}\psi^n-\frac{\Delta tI_f}{\epsilon_0\left[\epsilon_\infty+\chi^0\right]} \\
    &-\frac{1}{\epsilon_0\left[\epsilon_\infty+\chi^0\right]}\frac{\Delta t}{\Delta z}\left[H^{i+1,n}-H^{i,n}\right]

We then implement the Yee cell in the simulation by offsetting the electric and magnetic field cells by half a spatial
and temporal increment\cite{beard}, producing

.. math::

    H^{i+1/2,n+1/2}=&H^{i+1/2,n-1/2}-\frac{1}{\mu_0}\frac{\Delta t}{\Delta z}\left[E^{i+1,n}-E^{i,n}\right] \label{eq:hup} \\
    E^{i,n+1}=&\frac{\epsilon_\infty}{\epsilon_\infty+\chi^0}E^{i,n}+\frac{1}{\epsilon_\infty+\chi^0}\psi^n-\frac{\Delta tI_f}{\epsilon_0\left[\epsilon_\infty+\chi^0\right]} \\
    &-\frac{1}{\epsilon_0\left[\epsilon_\infty+\chi^0\right]}\frac{\Delta t}{\Delta z}\left[H^{i+1/2,n+1/2}-H^{i-1/2,n+1/2}\right]

where

.. math::

    \Delta\chi^m&=\chi^m-\chi^{m+1} \nonumber \\
    \psi^n&=\sum^{n-1}_{m=0}E^{i,n-m}\Delta\chi^m \label{eq:psi}

The accuracy of the derivative approximation inherent to these update equations relies on choosing some :math:`\Delta z`
and :math:`\Delta t` small enough such that the electric and magnetic fields are approximately linear over spatial
intervals :math:`\Delta z` and time intervals :math:`\Delta t`. If this condition is not met then the accuracy of the
derivative approximation breaks down. The update equations derived here are significant as they reveal that any linear
dielectric can be accurately simulated via the RC-FDTD method as long as the electric susceptibility of the material
:math:`\chi(t)` is well defined. We turn our attention to modeling the electric susceptibility of materials in section
\ref{sec:susceptibility}.

rcfdtdpy provides a framework in which the user need only provide the electric susceptibility :math:`\chi(t)` to run a
simulation.