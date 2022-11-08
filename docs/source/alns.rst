.. module:: alns
   :synopsis: ALNS


ALNS
====

The top-level :mod:`alns` module contains two public classes: :class:`~alns.ALNS.ALNS` and :class:`~alns.State.State`.
The first is used to run the ALNS algorithm.
The second can be subclassed to define a solution state: all that is needed is to define a `objective()` member function that returns the solution cost.

Finally, the :meth:`~alns.ALNS.ALNS.iterate` method on :class:`~alns.ALNS.ALNS` instances returns a :class:`~alns.Result.Result` object.
Its properties and methods can be used to access the final solution and runtime statistics.
The :class:`~alns.Statistics.Statistics` object is also presented below, but it is typically not necessary to interact with it: most things are already available via the :class:`~alns.Result.Result` object.

.. automodule:: alns.ALNS
   :members:

.. automodule:: alns.State
   :members:

.. automodule:: alns.Result
   :members:

.. automodule:: alns.Outcome
   :members:

.. automodule:: alns.Statistics
   :members:
