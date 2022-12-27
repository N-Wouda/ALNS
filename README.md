[![PyPI version](https://badge.fury.io/py/alns.svg)](https://badge.fury.io/py/alns)
[![ALNS](https://github.com/N-Wouda/ALNS/actions/workflows/alns.yaml/badge.svg)](https://github.com/N-Wouda/ALNS/actions/workflows/alns.yaml)
[![Documentation Status](https://readthedocs.org/projects/alns/badge/?version=latest)](https://alns.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/N-Wouda/ALNS/branch/master/graph/badge.svg)](https://codecov.io/gh/N-Wouda/ALNS)

``alns`` is a general, well-documented and tested implementation of the adaptive
large neighbourhood search (ALNS) metaheuristic in Python. The ALNS
metaheuristic is an algorithm that can be used to solve difficult combinatorial
optimisation problems. The algorithm begins with an initial solution. This
solution should be feasible, but need not be very good. Then the algorithm
iterates until a stopping criterion is met. In each iteration, it selects a
destroy and repair operator, which transform the current solution into a
candidate solution. This candidate solution is then evaluated by an acceptance
criterion, and the operator selection mechanism is updated based on the
evaluation outcome.

It may be installed in the usual way as

```
pip install alns
```

### Getting started

The `alns` library provides the ALNS algorithm and various acceptance criteria,
operator selection schemes, and stopping criteria. The available options are
further explained in the [documentation][1]. You should provide the following:

- A solution state for your problem that implements an `objective()` function.
- An initial feasible solution.
- One or more destroy and repair operators.

> A "quickstart" code template is available [here][10].

### Examples

The [documentation][1] contains example notebooks showing how the ALNS library
may be used. These include:

- The travelling salesman problem (TSP), [here][2]. We solve an instance of 131
  cities in one minute to a 2% optimality gap, using very simple destroy and
  repair heuristics.
- The capacitated vehicle routing problem (CVRP), [here][8]. We solve an
  instance with 241 customers to within 3% of optimality using a combination of
  a greedy repair operator, and a _slack-induced substring removal_ destroy
  operator.
- The cutting-stock problem (CSP), [here][4]. We solve an instance with 180
  beams over 165 distinct sizes to within 1.35% of optimality in only a very
  limited number of iterations.
- The resource-constrained project scheduling problem (RCPSP), [here][6]. We
  solve an instance with 90 jobs and 4 resources to within 4% of the best known
  solution, using a number of different operators and enhancement techniques
  from the literature.
- The permutation flow shop problem (PFSP), [here][9]. We solve an instance with
  50 jobs and 20 machines to within 1.5% of the best known solution. Moreover,
  we demonstrate multiple advanced features of ALNS, including auto-fitting the
  acceptance criterion and adding local search to repair operators. We also
  demonstrate how one could tune ALNS parameters.

Finally, the features notebook gives an overview of various options available in
the `alns` package. In the notebook we use these different options to solve a
toy 0/1-knapsack problem. The notebook is a good starting point for when you
want to use different schemes, acceptance or stopping criteria yourself. It is
available [here][5].

### Contributing

We are very grateful for any contributions you are willing to make. Please have
a look [here][3] to get started. If you aim to make a large change, it likely
helpful to discuss the change first in a new GitHub issue. Feel free to open
one!

### Getting help

If you are looking for help, please follow the instructions [here][7].

[1]: https://alns.readthedocs.io/en/latest/

[2]: https://alns.readthedocs.io/en/latest/examples/travelling_salesman_problem.html

[3]: https://alns.readthedocs.io/en/latest/setup/contributing.html

[4]: https://alns.readthedocs.io/en/latest/examples/cutting_stock_problem.html

[5]: https://alns.readthedocs.io/en/latest/examples/alns_features.html

[6]: https://alns.readthedocs.io/en/latest/examples/resource_constrained_project_scheduling_problem.html

[7]: https://alns.readthedocs.io/en/latest/setup/getting_help.html

[8]: https://alns.readthedocs.io/en/latest/examples/capacitated_vehicle_routing_problem.html

[9]: https://alns.readthedocs.io/en/latest/examples/permutation_flow_shop_problem.html

[10]: https://alns.readthedocs.io/en/latest/setup/template.html
