.. figure:: assets/images/logo.svg
   :alt: ALNS logo
   :figwidth: 100%

``alns`` is a general, well-documented and tested implementation of the adaptive large neighbourhood search (ALNS) metaheuristic in Python.
ALNS is an algorithm that can be used to solve difficult combinatorial optimisation problems.
The algorithm begins with an initial solution.
Then the algorithm iterates until a stopping criterion is met.
In each iteration, a destroy and repair operator are selected, which transform the current solution into a candidate solution.
This candidate solution is then evaluated by an acceptance criterion, and the operator selection scheme is updated based on the evaluation outcome.

It is common in the operations research community to re-implement heuristics such as ALNS.
Such implementations are relatively limited, are typically tied tightly to one particular problem domain, and often implement just a single acceptance criterion and operator selection scheme.
This inhibits experimentation with different aspects of the algorithm, and makes re-use by others or in other problem domains difficult.
This ``alns`` package, by contrast, offers a clear and problem-agnostic API for using the ALNS algorithm, and provides many acceptance criteria and operator selection schemes.
Additionally, we provide diagnostic statistics, plotting methods, logging, and the ability to register custom callbacks at various points of the search.
These allow researchers and practitioners to rapidly develop state-of-the-art metaheuristics in a wide range of problem domains.

The ``alns`` package depends only on ``numpy`` and ``matplotlib``.
It can be installed through *pip* via

.. code-block:: shell

   pip install alns

.. hint::

    Have a look at the :doc:`quickstart template <setup/template>` to get started on your own ALNS metaheuristic.
    If you are new to metaheuristics or ALNS, you might benefit from first reading the :doc:`introduction to ALNS <setup/introduction_to_alns>`.
    To set up an installation from source, or to run the examples listed below yourself, please have a look at the :doc:`installation instructions <setup/installation>`.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   setup/introduction_to_alns
   setup/template
   setup/installation
   setup/contributing
   setup/getting_help

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/alns_features
   examples/cutting_stock_problem
   examples/resource_constrained_project_scheduling_problem
   examples/travelling_salesman_problem
   examples/capacitated_vehicle_routing_problem
   examples/permutation_flow_shop_problem
   examples/other single-trajectory heuristics

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/alns
   api/accept
   api/select
   api/stop
