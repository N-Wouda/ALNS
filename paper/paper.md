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
ALNS has quickly become a favourite in the field of combinatorial optimisation and operations research [@Windras_Mara_et_al:2022] for solving difficult NP-hard problems.
These problems include many variants of the vehicle routing problem and various scheduling problems.
Our package has an easy-to-use API and includes various stopping criteria, a large set of acceptance criteria based on @Santini_et_al:2018, and multiple operator selection schemes.
Furthermore, it supports many other single-trajectory neighbourhood search algorithms as special cases, including iterated local search (ILS), variable neighbourhood descent (VND), and the greedy randomised adaptive search procedure (GRASP).
The package can be used for research into the application of these heuristic algorithms to various problems in industry TODO, or research into improving the metaheuristic itself TODO.

# Statement of need

TODO:

- State of the art
- Increase re-use, limit duplicate work (explain alternatives)
- Allow more experimentation with different ALNS aspects (since we already implemented them)
- Explain existing libraries are limited, poorly documented, and/or non-existent

# Features

The `alns` library is a Python package that offers:

- TODO
- TODO
- TODO

TODO documentation?

# References
