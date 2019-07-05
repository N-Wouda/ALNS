[![PyPI version](https://badge.fury.io/py/alns.svg)](https://badge.fury.io/py/alns)
[![Build Status](https://travis-ci.com/N-Wouda/ALNS.svg?branch=master)](https://travis-ci.com/N-Wouda/ALNS)
[![codecov](https://codecov.io/gh/N-Wouda/ALNS/branch/master/graph/badge.svg)](https://codecov.io/gh/N-Wouda/ALNS)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/0c573395b313403b96c26054973dec34)](https://www.codacy.com/app/N-Wouda/ALNS)

This package offers a general, well-documented and tested
implementation of the adaptive large neighbourhood search (ALNS)
meta-heuristic, based on the description given in [Pisinger and Ropke
(2010)][1]. It may be installed in the usual way as,

```
pip install alns
```

## How to use
The `alns` package exposes two classes, `ALNS` and `State`. The first
may be used to run the ALNS algorithm, the second may be subclassed to
store a solution state - all it requires is to define an `objective`
member function.

The ALNS algorithm must be supplied with an acceptance criterion, to
determine the acceptance of a new solution state at each iteration.
An overview of common acceptance criteria is given in [Santini et al.
(2018)][3]. Several have already been implemented for you, in
`alns.criteria`,

- `HillClimbing`. The simplest acceptance criterion, hill-climbing
  solely accepts solutions improving the objective value.
- `RecordToRecordTravel`. This criterion only accepts solutions when
  the improvement meets some updating threshold.
- `SimulatedAnnealing`. This criterion accepts solutions when the
  scaled probability is bigger than some random number, using an
  updating temperature.

Each acceptance criterion inherits from `AcceptanceCriterion`, which may
be used to write your own.

### Examples
The `examples/` directory features some example notebooks showcasing
how the ALNS library may be used. Of particular interest are,

- The travelling salesman problem (TSP), [here][2]. We solve an
  instance of 131 cities to within 2.1% of optimality, using simple
  destroy and repair heuristics with a post-processing step.

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
