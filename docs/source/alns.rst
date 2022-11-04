.. module:: alns
   :synopsis: ALNS


ALNS
====

The top-level :mod:`alns` module contains two public classes: :class:`~alns.ALNS` and :class:`~alns.State`.
The first is used to run the ALNS algorithm.
The second can be subclassed to define a solution state: all that is needed is to define a `objective()` member function that returns the solution cost.

.. currentmodule:: alns

.. autosummary::
   :toctree: generated/

   ALNS
   State
