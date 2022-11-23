ALNS
====

**ALNS** is a general, well-documented and tested implementation of the adaptive large neighbourhood search (ALNS) meta-heuristic in Python.

It is common in the operations research community to re-implement heuristics such as ALNS.
Such implementations are relatively limited, are typically tied tightly to one particular problem domain, and often implement just a single acceptance criterion and operator selection scheme.
This inhibits experimentation with different aspects of the algorithm, and makes re-use by others or in other problem domains difficult.
This ``alns`` package, by contrast, offers a clear and problem-agnostic API for using the ALNS algorithm, and provides many acceptance criteria and operator selection schemes.
Additionally, we provide diagnostic statistics, plotting methods, logging, and the ability to register custom callbacks at various points of the search.
These allow researchers and practitioners to rapidly develop state-of-the-art metaheuristics in a wide range of problem domains.

ALNS can be installed through *pip* via

.. code-block:: shell

   pip install alns

Getting started
---------------

.. toctree::
   :maxdepth: 1

   setup/installation
   setup/contributing

Examples
--------

The following pages show several complete, real-world examples of ALNS metaheuristics, implemented using the ``alns`` package.
These examples can be used to quickstart your own metaheuristic.
To run these examples yourself, please have a look at the :doc:`setup/installation`.

.. toctree::
   :maxdepth: 1

   examples/alns_features
   examples/cutting_stock_problem
   examples/resource_constrained_project_scheduling_problem
   examples/travelling_salesman_problem
   examples/capacitated_vehicle_routing_problem
   examples/permutation_flow_shop_problem
   examples/other single-trajectory heuristics

API reference
-------------

.. toctree::
   :maxdepth: 1

   api/alns
   api/accept
   api/select
   api/stop
