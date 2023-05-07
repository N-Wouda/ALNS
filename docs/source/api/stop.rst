.. module:: alns.stop
   :synopsis: Stopping criteria


Stopping criteria
=================

The :mod:`alns.stop` module contains the various stopping criteria the `alns` package ships with.
These can be used to stop the ALNS search whenever some criterion is met: for example, when some maximum number of iterations or run-time is exceeded.

All stopping criteria implement :class:`~alns.stop.StoppingCriterion.StoppingCriterion`.

.. automodule:: alns.stop.StoppingCriterion
   :members:

.. automodule:: alns.stop.MaxIterations
   :members:

.. automodule:: alns.stop.MaxRuntime
   :members:

.. automodule:: alns.stop.NoImprovement
   :members:
