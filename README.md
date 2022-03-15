[![PyPI version](https://badge.fury.io/py/alns.svg)](https://badge.fury.io/py/alns)
[![ALNS](https://github.com/N-Wouda/ALNS/actions/workflows/alns.yml/badge.svg)](https://github.com/N-Wouda/ALNS/actions/workflows/alns.yml)
[![codecov](https://codecov.io/gh/N-Wouda/ALNS/branch/master/graph/badge.svg)](https://codecov.io/gh/N-Wouda/ALNS)

This package offers a general, well-documented and tested
implementation of the adaptive large neighbourhood search (ALNS)
meta-heuristic, based on the description given in [Pisinger and Ropke
(2010)][1]. It may be installed in the usual way as
```
pip install alns
```

### Examples
If you wish to dive right in, the `examples/` directory contains example notebooks
showing how the ALNS library may be used. These include:

- The travelling salesman problem (TSP), [here][2]. We solve an
  instance of 131 cities to within 2.1% of optimality, using simple
  destroy and repair heuristics with a post-processing step.
- The cutting-stock problem (CSP), [here][4]. We solve an instance with
  180 beams over 165 distinct sizes to within 1.35% of optimality in
  only a very limited number of iterations.
- The resource-constrained project scheduling problem, [here][6]. We solve an
  instance with 90 jobs and 4 resources to within 4% of the best known solution,
  using a number of different operators and enhancement techniques from the 
  literature.

Finally, the weight schemes and acceptance criteria notebook gives an overview
of various options available in the `alns` package (explained below). In the
notebook we use these different options to solve a toy 0/1-knapsack problem. The
notebook is a good starting point for when you want to use the different schemes
and criteria yourself. It is available [here][5].

## How to use
The `alns` package exposes two classes, `ALNS` and `State`. The first
may be used to run the ALNS algorithm, the second may be subclassed to
store a solution state - all it requires is to define an `objective`
member function, returning an objective value.

The ALNS algorithm must be supplied with a _weight scheme_ and an _acceptance
criterion_.

### Weight scheme
The weight scheme determines how to select destroy and repair operators in each
iteration of the ALNS algorithm. Several have already been implemented for you,
in `alns.weight_schemes`:

- `SimpleWeights`. This weight scheme applies a convex combination of the 
   existing weight vector, and a reward given for the current candidate 
   solution.
- `SegmentedWeights`. This weight scheme divides the iteration horizon into
   segments. In each segment, scores are summed for each operator. At the end
   of each segment, the weight vector is updated as a convex combination of 
   the existing weight vector, and these summed scores.

Each weight scheme inherits from `WeightScheme`, which may be used to write 
your own.

### Acceptance criterion
The acceptance criterion determines the acceptance of a new solution state at
each iteration. An overview of common acceptance criteria is given in
[Santini et al. (2018)][3]. Several have already been implemented for you, in
`alns.criteria`:

- `HillClimbing`. The simplest acceptance criterion, hill-climbing solely
  accepts solutions improving the objective value.
- `RecordToRecordTravel`. This criterion accepts solutions when the improvement
  meets some updating threshold.
- `SimulatedAnnealing`. This criterion accepts solutions when the
  scaled probability is bigger than some random number, using an
  updating temperature.

Each acceptance criterion inherits from `AcceptanceCriterion`, which may
be used to write your own.

## References
- Pisinger, D., and Ropke, S. (2010). Large Neighborhood Search. In M.
  Gendreau (Ed.), _Handbook of Metaheuristics_ (2 ed., pp. 399-420).
  Springer.
- Santini, A., Ropke, S. & Hvattum, L.M. (2018). A comparison of
  acceptance criteria for the adaptive large neighbourhood search
  metaheuristic. *Journal of Heuristics* 24 (5): 783-815.

[1]: http://orbit.dtu.dk/en/publications/large-neighborhood-search(61a1b7ca-4bf7-4355-96ba-03fcdf021f8f).html
[2]: https://github.com/N-Wouda/ALNS/blob/master/examples/travelling_salesman_problem.ipynb
[3]: https://link.springer.com/article/10.1007%2Fs10732-018-9377-x
[4]: https://github.com/N-Wouda/ALNS/blob/master/examples/cutting_stock_problem.ipynb
[5]: https://github.com/N-Wouda/ALNS/blob/master/examples/weight_schemes_acceptance_criteria.ipynb
[6]: https://github.com/N-Wouda/ALNS/blob/master/examples/resource_constrained_project_scheduling_problem.ipynb
