.. module:: alns.select
   :synopsis: Operator selection schemes


Operator selection schemes
==========================

The :mod:`alns.select` module contains the various operator selection schemes the `alns` package ships with.
These are used during the ALNS search to select a destroy and repair operator pair in each iteration.

All stopping criteria inherit from :class:`~alns.select.SelectionScheme`, which can be subclassed to create your own.

.. currentmodule:: alns.select

.. autosummary::
   :toctree: generated/

   SelectionScheme
   RouletteWheel
   SegmentedRouletteWheel
