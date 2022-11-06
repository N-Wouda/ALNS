Other single-trajectory heuristics
==================================

The ``alns`` package supports several other single-trajectory heuristics as special cases, in addition to 'just' ALNS.
This page explains how to implement iterated local search (ILS), variable neighbourhood search (VNS), and the greedy randomised adaptive search procedure (GRASP) using ``alns``.


ILS
---

`ILS <https://en.wikipedia.org/wiki/Iterated_local_search>`_ is an iterative algorithm that, in each iteration, performs two things:

1. Perturb the current solution;
2. Perform local search on the perturbed solution.

ILS is easy to implement in ``alns`` by having one destroy operator that is responsible for the perturbation, and one repair operator that performs the local search.
Since there is just one destroy and repair operator pair, the operator selection scheme is not relevant.
We suggest to use the simplest scheme: :class:`~alns.select.RouletteWheel`.
At a high level, one could thus implement the following:

.. code-block:: python

   from alns import ALNS, State


   def perturb(sol: State, rnd_state) -> State:
       <perturb sol>
       return <perturbed solution>


   def local_search(sol: State, rnd_state) -> State:
       <perform local search around sol>
       return <improved solution>


   alns = ALNS()
   alns.add_destroy_operator(perturb)
   alns.add_repair_operator(local_search)

Where the choice of acceptance and stopping criterion are left to the user.


VNS
---

`VNS <https://en.wikipedia.org/wiki/Variable_neighborhood_search>`_ is an iterative algorithm that, in each iteration, performs the following steps:

1. Perturb the current solution;
2. Perform local search on the perturbed solution;
3. Possibly change neighbourhoods.

TODO

GRASP
-----

`GRASP <https://en.wikipedia.org/wiki/Greedy_randomized_adaptive_search_procedure>`_ is an iterative algorithm that performs a greedy randomised improvement step in each iteration.
This greedy randomised step could start from an empty solution, or from a partial solution.
This suggests one destroy operator that is responsible for either generating an empty solution, or a partially broken solution that can be repaired by a greedy randomised repair operator.
At a high level, one could thus implement the following:

.. code-block:: python

   from alns import ALNS, State


   def destroy(sol: State, rnd_state) -> State:
       <destroy sol to some fixed degree of destruction (possibly completely)>
       return <destroyed solution>


   def greedy_randomised_repair(sol: State, rnd_state) -> State:
       <do greedy randomised repair around sol>
       return <improved solution>


   alns = ALNS()
   alns.add_destroy_operator(destroy)
   alns.add_repair_operator(greedy_randomised_repair)

We again suggest to use :class:`~alns.select.RouletteWheel`, and leave the choice of acceptance and stopping criterion are left to the user.
