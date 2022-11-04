.. module:: alns.accept
   :synopsis: Acceptance criteria


Acceptance criteria
===================

The :mod:`alns.accept` module contains the various acceptance criteria the `alns` package ships with.
These criteria are used by the ALNS algorithm to decide whether to accept or reject a candidate solution.

All acceptance criteria inherit from :class:`~alns.accept.AcceptanceCriterion`, which can be subclassed to create your own.

.. currentmodule:: alns.accept

.. autosummary::
   :toctree: generated/

   AcceptanceCriterion
   GreatDeluge
   NonLinearGreatDeluge
   HillClimbing
   LateAcceptanceHillClimbing
   RandomWalk
   RecordToRecordTravel
   SimulatedAnnealing
   WorseAccept
