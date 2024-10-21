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


   def perturb(sol: State, rng) -> State:
       <perturb sol>
       return <perturbed solution>


   def local_search(sol: State, rng) -> State:
       <perform local search around sol>
       return <improved solution>


   alns = ALNS()
   alns.add_destroy_operator(perturb)
   alns.add_repair_operator(local_search)

Where the choice of acceptance and stopping criterion are left to the user.


VNS
---

`VNS <https://en.wikipedia.org/wiki/Variable_neighborhood_search>`_ is an iterative algorithm that, in each iteration, performs the following steps given a neighbourhood :math:`\mathcal{N}_k`:

1. Perturb the current solution (possibly using :math:`\mathcal{N}_k`);
2. Perform local search in :math:`\mathcal{N}_k` on the perturbed solution;
3. Change neighbourhoods.

The first two steps look a lot like ILS.
For the third, we need a bit more: an object to store :math:`k` and a list of neighbourhoods.
Assume we have this list of neighbourhoods available.
Then, a high-level implementation could look like:

.. code-block:: python

   from dataclasses import dataclass

   from alns import ALNS, State


   @dataclass
   class Neighbourhood:
       neighbourhoods: list
       k: int


   def perturb(sol: State, rng, neighbourhood: Neighbourhood) -> State:
       <perturb sol, possibly using neighbourhood k>
       return <perturbed solution>


   def local_search(
       sol: State,
       rng,
       neighbourhood: Neighbourhood
   ) -> State:
       <perform local search around sol using neighbourhood k>

       # Set next neighbourhood: if we found an improving solution, the
       # callback will reset the neighbourhood; else we start from the next
       # neighbourhood in the following iteration.
       neighbourhood.k = min(
           neighbourhood.k + 1,
           len(neighbourhood.neighbourhoods)
       )

       return <improved solution>


   def on_best(sol: State, rng, neighbourhood: Neighbourhood):
       # New best solution: start again from first neighbourhood.
       neighbourhood.k = 1


   neighbourhood = Neighbourhood(<neighbourhoods>, 1)
   alns = ALNS()
   alns.on_best(on_best)
   alns.add_destroy_operator(perturb)
   alns.add_repair_operator(local_search)

   res = alns.iterate(..., neighbourhood=neighbourhood)


This example uses two somewhat advanced features: first, we use the :meth:`~alns.ALNS.ALNS.on_best` callback function to reset the neighbourhoods in case of improvement.
Second, we use the flexible ``**kwargs`` argument of :meth:`~alns.ALNS.ALNS.iterate` to pass the ``neighbourhood`` object to the operators.

We again suggest to use :class:`~alns.select.RouletteWheel`, and leave the choice of acceptance and stopping criterion to the user.


GRASP
-----

`GRASP <https://en.wikipedia.org/wiki/Greedy_randomized_adaptive_search_procedure>`_ is an iterative algorithm that performs a greedy randomised improvement step in each iteration.
This greedy randomised step could start from an empty solution, or from a partial solution.
This suggests one destroy operator that is responsible for either generating an empty solution, or a partially broken solution that can be repaired by a greedy randomised repair operator.
At a high level, one could thus implement the following:

.. code-block:: python

   from alns import ALNS, State


   def destroy(sol: State, rng) -> State:
       <destroy sol to some fixed degree of destruction (possibly completely)>
       return <destroyed solution>


   def greedy_randomised_repair(sol: State, rng) -> State:
       <do greedy randomised repair around sol>
       return <improved solution>


   alns = ALNS()
   alns.add_destroy_operator(destroy)
   alns.add_repair_operator(greedy_randomised_repair)

We again suggest to use :class:`~alns.select.RouletteWheel`, and leave the choice of acceptance and stopping criterion to the user.
