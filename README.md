[![PyPI version](https://badge.fury.io/py/alns.svg)](https://badge.fury.io/py/alns)
[![ALNS](https://github.com/N-Wouda/ALNS/actions/workflows/alns.yaml/badge.svg)](https://github.com/N-Wouda/ALNS/actions/workflows/alns.yaml)
[![Documentation Status](https://readthedocs.org/projects/alns/badge/?version=latest)](https://alns.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/N-Wouda/ALNS/branch/master/graph/badge.svg)](https://codecov.io/gh/N-Wouda/ALNS)

This package offers a general, well-documented and tested
implementation of the adaptive large neighbourhood search (ALNS)
meta-heuristic, based on the description given in [Pisinger and Ropke
(2010)][1]. It may be installed in the usual way as
```
pip install alns
```

### Examples
If you wish to dive right in, the [documentation][7] contains example notebooks
showing how the ALNS library may be used. These include:

- The travelling salesman problem (TSP), [here][2]. We solve an instance of 131
  cities in one minute to a 2% optimality gap, using very simple destroy and
  repair heuristics.
- The capacitated vehicle routing problem (CVRP), [here][8]. We solve an 
  instance with 241 customers to within 3% of optimality using a combination
  of a greedy repair operator, and a _slack-induced substring removal_ destroy
  operator.
- The cutting-stock problem (CSP), [here][4]. We solve an instance with
  180 beams over 165 distinct sizes to within 1.35% of optimality in
  only a very limited number of iterations.
- The resource-constrained project scheduling problem (RCPSP), [here][6]. We solve 
  an instance with 90 jobs and 4 resources to within 4% of the best known solution,
  using a number of different operators and enhancement techniques from the 
  literature.
- The permutation flow shop problem (PFSP), [here][9]. We solve an instance with
  50 jobs and 20 machines to within 1.5% of the best known solution. Moreover,
  we demonstrate multiple advanced features of ALNS, including auto-fitting the
  acceptance criterion and adding local search to repair operators. We also
  demonstrate how one could tune ALNS parameters.

Finally, the features notebook gives an overview of various options available 
in the `alns` package. In the notebook we use these different options to solve
a toy 0/1-knapsack problem. The notebook is a good starting point for when you
want to use different schemes, acceptance or stopping criteria yourself. It is
available [here][5].

## How to use
Our [documentation][7] provides a complete overview of the `alns` package. In 
short: the `alns` package exposes two classes, `ALNS` and `State`. The first
may be used to run the ALNS algorithm, the second may be subclassed to
store a solution state - all it requires is to define an `objective`
member function, returning an objective value.

The ALNS algorithm must be supplied with an _operator selection scheme_, an
_acceptance criterion_, and a _stopping criterion_. These are explained further
in the documentation.

## References
- Pisinger, D., and Ropke, S. (2010). Large Neighborhood Search. In M.
  Gendreau (Ed.), _Handbook of Metaheuristics_ (2 ed., pp. 399-420).
  Springer.
- Santini, A., Ropke, S. & Hvattum, L.M. (2018). A comparison of
  acceptance criteria for the adaptive large neighbourhood search
  metaheuristic. *Journal of Heuristics* 24 (5): 783-815.

[1]: http://orbit.dtu.dk/en/publications/large-neighborhood-search(61a1b7ca-4bf7-4355-96ba-03fcdf021f8f).html
[2]: https://alns.readthedocs.io/en/latest/examples/travelling_salesman_problem.html
[3]: https://link.springer.com/article/10.1007%2Fs10732-018-9377-x
[4]: https://alns.readthedocs.io/en/latest/examples/cutting_stock_problem.html
[5]: https://alns.readthedocs.io/en/latest/examples/alns_features.html
[6]: https://alns.readthedocs.io/en/latest/examples/resource_constrained_project_scheduling_problem.html
[7]: https://alns.readthedocs.io/en/latest/
[8]: https://alns.readthedocs.io/en/latest/examples/capacitated_vehicle_routing_problem.html
[9]: https://alns.readthedocs.io/en/latest/examples/permutation_flow_shop_problem.html
