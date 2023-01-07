A brief introduction to ALNS
============================

Before we explain what ALNS is, we first need to describe the type of problems it can solve well.
These problems typically come from the field of `combinatorial optimisation <https://en.wikipedia.org/wiki/Combinatorial_optimization>`_, and have

* A large (but finite) set of solutions (the *search space*)
* An objective function to be optimised by finding a *good* solution in the search space

A very well-known example of such a problem is the `travelling salesman problem <https://en.wikipedia.org/wiki/Travelling_salesman_problem>`_ (TSP), which asks

    Given a list of :math:`n` cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?

It is easy to find *a* solution: :math:`1 \rightarrow 2 \rightarrow 3 \rightarrow \ldots \rightarrow n \rightarrow 1` is one option.
Finding an *optimal* solution, however, can be very difficult since there are :math:`n!` possible solutions, and the search space thus grows very quickly with :math:`n`.

.. note::

    :ref:`This example notebook </examples/travelling_salesman_problem.ipynb>` solves a TSP instance using ALNS.

Due to the explosive growth of the search space, most combinatorial problems cannot be solved optimally by testing every solution in the search space.
For these problems, a heuristic method can help find good (but not necessarily optimal) solutions quickly.
The adaptive large neighbourhood search (ALNS) algorithm is one such heuristic method.
Specifically, ALNS is:

* A large neighbourhood search (LNS) method.
  LNS methods explore large subsets of the search space in a systematic manner.

* A *meta* heuristic.
  Metaheuristics use other heuristics as operators.
  ALNS, in particular, is a ruin-and-recreate algorithm: it relies on heuristic *destroy* and *repair* operators to explore the search space.

* Adaptive.
  During the search, ALNS learns about operators that are more effective than others, and uses that information to choose those effective operators more often.

.. note::

    For a more thorough introduction to LNS and LNS-based metaheuristics like ALNS, the handbook chapter of `Pisinger and Røpke (2019) <https://doi.org/10.1007/978-3-319-91086-4_4>`_ may be useful.

ALNS begins with an initial solution and then iterates until a stopping criterion is met.
In each iteration, a destroy and repair operator are selected, which transform the current solution into a candidate solution.
This candidate solution is then evaluated by an acceptance criterion, and the operator selection scheme is updated based on the evaluation outcome.
Once the stopping criterion is met, ALNS returns the best solution it has found.
In pseudocode, ALNS works as follows:

    .. line-block::

        **Input:** initial solution :math:`s`
        **Output:** optimised solution :math:`s^*`
        :math:`s^* \gets s`
        **repeat** until stopping criteria is met:
            Select destroy and repair operator pair :math:`(d, r)` using operator selection scheme
            :math:`s^c \gets r(d(s))`
            **if** :math:`s^c` is accepted:
                :math:`s \gets s^c`
            **if** :math:`s^c` has a better objective value than :math:`s^*`:
                :math:`s^* \gets s^c`
            Update operator selection scheme
        **return** :math:`s^*`

The ``alns`` package provides the ALNS algorithm, stopping and acceptance criteria, and various operator selection schemes for you.
You need to provide an initial solution, and one or more destroy and repair operators tailored to your problem.

.. hint::

    The :doc:`quickstart code template <template>` explains how to implement these in the format ``alns`` expects.
