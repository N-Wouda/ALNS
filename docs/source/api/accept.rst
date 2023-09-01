.. module:: alns.accept
   :synopsis: Acceptance criteria


Acceptance criteria
===================

The :mod:`alns.accept` module contains the various acceptance criteria the `alns` package ships with.
These criteria are used by the ALNS algorithm to decide whether to accept or reject a candidate solution.

All acceptance criteria implement :class:`~alns.accept.AcceptanceCriterion.AcceptanceCriterion`.

.. automodule:: alns.accept.AcceptanceCriterion
   :members:

.. automodule:: alns.accept.AlwaysAccept
   :members:

.. automodule:: alns.accept.GreatDeluge
   :members:

.. automodule:: alns.accept.HillClimbing
   :members:

.. automodule:: alns.accept.LateAcceptanceHillClimbing
   :members:

.. automodule:: alns.accept.MovingAverageThreshold
   :members:

.. automodule:: alns.accept.NonLinearGreatDeluge
   :members:

.. automodule:: alns.accept.RandomAccept
   :members:

.. automodule:: alns.accept.RecordToRecordTravel
   :members:

.. automodule:: alns.accept.SimulatedAnnealing
   :members:
