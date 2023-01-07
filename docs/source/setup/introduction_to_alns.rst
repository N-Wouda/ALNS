A brief introduction to ALNS
============================

Before we explain what ALNS is, we first need to describe the type of problems it can solve well.
These problems typically come from the field of `combinatorial optimisation <https://en.wikipedia.org/wiki/Combinatorial_optimization>`_, and have

* A large (but finite) set of solutions (the *search space*)
* An objective to be optimised by finding a *good* solution in the search space

A very well-known example of such a problem is the `travelling salesman problem <https://en.wikipedia.org/wiki/Travelling_salesman_problem>`_ (TSP), which asks

    Given a list of :math:`n` cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?

It is easy to find *a* solution: :math:`1 \rightarrow 2 \rightarrow 3 \rightarrow \ldots \rightarrow n \rightarrow 1` is one option.
Finding an *optimal* solution, however, can be very difficult since there are :math:`n!` possible solutions, and the search space thus grows very quickly with :math:`n`.

Because of the explosive growth of the search space, combinatorial problems cannot be solved optimally by testing every solution in the search space.
In such cases, a heuristic method can help find good (but not necessarily optimal) solutions quickly.
ALNS is one such heuristic method.

.. note::

    :ref:`/examples/travelling_salesman_problem.ipynb` solves a TSP instance using ALNS.

Specifically, the adaptive large neighbourhood search metaheuristic is:

* A large neighbourhood search (LNS) method.
  LNS methods explore large subsets of the search space in a systematic manner.

* A *meta* heuristic.
  Metaheuristics use other heuristics as operators.
  ALNS, in particular, is a ruin-and-recreate algorithm: it relies on heuristic *destroy* and *repair* operators to explore the search space.

* Adaptive.
  During the search, ALNS learns about operators that are more effective than others, and uses that information to choose those effective operators more often.

ALNS begins with an initial solution and then iterates until a stopping criterion is met.
In each iteration, a destroy and repair operator are selected, which transform the current solution into a candidate solution.
This candidate solution is then evaluated by an acceptance criterion, and the operator selection scheme is updated based on the evaluation outcome.
In pseudocode, ALNS works as follows:

    .. line-block::

        **Input:** an initial solution :math:`s`
        **Output:** optimised solution :math:`s^*`
        :math:`s^* \gets s`
        **repeat** until stopping criteria is met:
            Select destroy and repair operator pair :math:`(d, r)` using operator selection scheme
            :math:`s^c \gets r(d(s))`
            **if** candidate is accepted:
                :math:`s \gets s^c`
            **if** :math:`s^c` has a better objective value than :math:`s^*`:
                :math:`s^* \gets s^c`
            Update operator selection scheme
        **return** :math:`s^*`

The ``alns`` package provides the ALNS algorithm, stopping and acceptance criteria, and various operator selection schemes for you.
You need to provide:

* An initial solution.
* One or more destroy and repair operators tailored to your problem.

Typically, a good start for a destroy operator is *random removal*, which randomly destroys some part of the current solution.
A good repair operator is *greedy repair*, which repairs the partially destroyed solution in a greedy manner.

.. hint::

    Now that you know a little more about the ideas behind ALNS, the :doc:`quickstart code template <template>` is a great place to get started using the ``alns`` package!
