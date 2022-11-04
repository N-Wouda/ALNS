.. module:: alns.stop
   :synopsis: Stopping criteria


Stopping criteria
=================

The :mod:`alns.stop` module contains the various stopping criteria the `alns` package ships with.
These can be used to stop the ALNS search whenever some criterion is met: for example, when some maximum number of iterations or run-time is exceeded.

All stopping criteria inherit from :class:`~alns.stop.StoppingCriterion`, which can be subclassed to create your own.

.. currentmodule:: alns.stop

.. autosummary::
   :toctree: generated/

   StoppingCriterion
   MaxIterations
   MaxRuntime
   NoImprovement
