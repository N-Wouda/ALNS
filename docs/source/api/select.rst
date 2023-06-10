.. module:: alns.select
   :synopsis: Operator selection schemes


Operator selection schemes
==========================

The :mod:`alns.select` module contains the various operator selection schemes the `alns` package ships with.
These are used during the ALNS search to select a destroy and repair operator pair in each iteration.

All operator selection schemes inherit from :class:`~alns.select.OperatorSelectionScheme.OperatorSelectionScheme`.

.. automodule:: alns.select.OperatorSelectionScheme
   :members:

.. automodule:: alns.select.AlphaUCB
   :members:

.. automodule:: alns.select.MABSelector

   .. autoclass:: MABSelector
      :members:

.. automodule:: alns.select.RandomSelect
   :members:

.. automodule:: alns.select.RouletteWheel
   :members:

.. automodule:: alns.select.SegmentedRouletteWheel
   :members:
