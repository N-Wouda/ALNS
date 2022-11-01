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

The adaptive large neighbourhood search (ALNS) metaheuristic algorithm, introduced by @Ropke_Pisinger:2006, has quickly become a favourite in the field of combinatorial optimisation.
At its core, ALNS is an iterative ruin-and-recreate algorithm that runs until some stopping criterion is met.
The algorithm starts with some initial solution.
In each iteration, the current solution is transformed into a new candidate solution by the application of a destroy and repair operator.
These operators are typically problem-specific.
The candidate solution is then evaluated by an acceptance criterion, and possibly replaces the current solution.
Based on the outcome of that evaluation, the likelihood that some operators are selected is modified - this is the adaptive part of the metaheuristic.

Our `alns` package provides a library containing a complete implementation of ALNS as described in @Pisinger_Ropke:2010.
It additionally includes multiple stopping criteria, a large set of acceptance criteria based on @Santini_et_al:2018, and multiple operator selection schemes. 
Furthermore, it supports many other single-trajectory neighbourhood search algorithms as special cases, including iterated local search (ILS), variable neighbourhood descent (VND), and the greedy randomised adaptive search procedure (GRASP).
The package can thus be used for research into heuristic algorithms for various problems in industry, including TODO, TODO, and TODO.
Additionally, it can be used for research into improving the ALNS metaheuristic themselves, for example by improving operator selection using machine learning methods TODO.

# Statement of need

TODO

# Features

The `alns` library is a Python package that offers:

- TODO
- TODO
- TODO

# Installation and usage

TODO

# References
