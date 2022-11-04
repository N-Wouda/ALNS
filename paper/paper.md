---
title: 'ALNS: a flexible Python implementation of the adaptive large neighbourhood search metaheuristic'
tags:
  - Python
  - operations research
  - metaheuristics
  - adaptive large neighbourhood search
authors:
  - name: Niels A. Wouda
    orcid: 0000-0003-2463-0309
    corresponding: true
    affiliation: 1
  - name: Leon Lan
    orcid: 0000-0001-7479-0218
    affiliation: 2
affiliations:
 - name: "Department of Operations, University of Groningen, Groningen, The Netherlands \\newline"
   index: 1
 - name: "Department of Mathematics, Vrije Universiteit Amsterdam, Amsterdam, The Netherlands \\newline"
   index: 2
date: 1 November 2022
bibliography: paper.bib
---

# Summary

The `alns` Python package provides a complete implementation of the adaptive large neighbourhood search (ALNS) metaheuristic algorithm [@Ropke_Pisinger:2006; @Pisinger_Ropke:2010].
ALNS has quickly become a favourite in the field of operations research [@Windras_Mara_et_al:2022] for solving difficult combinatorial problems, including the vehicle routing problem, and various scheduling problems.
Our package has an easy-to-use API and includes various stopping criteria, a large set of acceptance criteria based on @Santini_et_al:2018, and multiple operator selection schemes.
Furthermore, it supports many other single-trajectory neighbourhood search algorithms as special cases, including iterated local search (ILS), variable neighbourhood descent (VND), and the greedy randomised adaptive search procedure (GRASP).
The package has already been used for research into further improvement of these heuristics [@Reijnen_et_al:2022] and the development of a high-quality heuristic for an industry problem [@Wouda_et_al:2022].
As such, we expect the package to be useful to the wider operations research community.

# Statement of need

It is common in the operations research community for researchers to implement heuristics from scratch [@vidal:2022].
This results in significant extra work for each researcher, and typically results in implementations that are relatively limited, often containing just enough to solve the researcher's problem well.
Such an implementation, for example, often only implements a single acceptance criterion and operator selection scheme, thus limiting the freedom to experiment with these aspects of the algorithm, and is often tied quite tightly to the researcher's problem domain.
This makes re-use difficult.
Our `alns` package, by contrast, offers a clear and problem-agnostic API for using the ALNS algorithm, including various acceptance and stopping criteria, and operator selection schemes.
Additionally, we provide diagnostic statistics, plotting methods, logging, and the ability to register custom callbacks at various points of the search.
These allow for rapid development of state-of-the-art metaheuristics in a wide range of problem domains.

TODO explain existing libraries are limited, poorly documented, and/or non-existent

# Features

The `alns` library is a Python package that offers:

- TODO
- TODO
- TODO

TODO documentation?

# References
