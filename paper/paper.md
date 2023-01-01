---
title: 'ALNS: a Python implementation of the adaptive large neighbourhood search metaheuristic'
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
date: 31 December 2022
bibliography: paper.bib
---

# Summary

The `alns` Python package provides a complete implementation of the adaptive large neighbourhood search (ALNS) metaheuristic algorithm [@Ropke_Pisinger:2006; @Pisinger_Ropke:2010].
ALNS has quickly become a favourite in the field of operations research for solving difficult combinatorial problems, including the vehicle routing problem and various scheduling problems.
Our package has an easy-to-use API and includes various stopping criteria, a large set of acceptance criteria based on @Santini_et_al:2018, and multiple operator selection schemes.
Furthermore, it supports many other single-trajectory neighbourhood search algorithms as special cases, including iterated local search (ILS), variable neighbourhood search (VNS), and the greedy randomised adaptive search procedure (GRASP).
The package has already been successfully used for research into methodological improvements of ALNS itself [@Reijnen_et_al:2022], and for the development of a high-quality ALNS metaheuristic solving an industry problem [@Wouda_et_al:2023].
Because of this success, we expect the package to be useful to the wider operations research community.

# Statement of need

Several software libraries exist that facilitate the implementation of metaheuristics [@Parejo2012-MetaheuristicOptimizationFrameworks].
The most widely-used libraries generally focus on population-based evolutionary algorithms and multi-objective optimization, for example `DEAP` [@Fortin2012-DEAPEvolutionaryAlgorithms], `ECJ` [@Scott2019-ECJ20General], `jMetal` [@Durillo2011-JMetalJavaFramework], `Metaheuristics.jl` [@Mejia-de-Dios2022-MetaheuristicsJuliaPackage], and `PySwarms` [@Miranda2018-PySwarmsResearchToolkit].
As such, these libraries provide limited functionality for single-trajectory algorithms like ALNS.
Among libraries that focus on single-trajectory algorithms are `Paradiseo` [@Dreo2021-ParadiseoModularFramework], and `Chips-n-Salsa` [@Cicirello2020-ChipsnSalsaJavaLibrary].
Neither provides an implementation of the ALNS algorithm out of the box.
Finally, the ALNS library by @Santini2019-AdaptiveLargeNeighbourhood is a problem-agnostic implementation of ALNS in C++.
This library provides a number of acceptance criteria, but supports only the roulette wheel operator selection scheme of @Ropke_Pisinger:2006.

Despite the existence of these software libraries, it remains common in the operations research community to re-implement heuristics [@Swan_et_al:2022].
Such implementations are relatively limited, are typically tied tightly to one particular problem domain, and often implement just a single acceptance criterion and operator selection scheme.
The survey of @Windras_Mara_et_al:2022 corroborates these claims: 205 out of the 251 papers they survey only consider a simulated annealing acceptance criterion, and only one paper uses an operator selection scheme that is not based on the roulette wheel mechanism of @Ropke_Pisinger:2006.
This inhibits experimentation with different aspects of the algorithm, and makes re-use by others or in other problem domains difficult.
Our `alns` package, by contrast, offers a clear and problem-agnostic API for using the ALNS algorithm, and provides many acceptance criteria and operator selection schemes.
Additionally, we provide diagnostic statistics, plotting methods, logging, and the ability to register custom callbacks at various points of the search.
These allow researchers and practitioners to rapidly develop state-of-the-art metaheuristics in a wide range of problem domains.

# Features

At its core, ALNS is an iterative ruin-and-recreate algorithm that runs until some stopping criterion is met.
The algorithm starts with some initial solution.
In each iteration, the current solution is transformed into a new candidate solution using problem-specific destroy and repair operators, which are selected via an operator selection scheme.
The candidate solution is then evaluated by an acceptance criterion, and possibly replaces the current solution.
Based on the outcome of that evaluation, the operator selection scheme updates the likelihood that the applied operators are selected again in the next iteration.

The `alns` Python package offers:

- A complete ALNS implementation, supported by an extensive test suite. 
  This implementation supports user-defined callbacks whenever a new solution is found, including when that new solution is a new global best, which could be used to support additional intensification methods.
  Furthermore, it can be supplied with arbitrary user-defined destroy and repair operators that are tailored to the user's problem domain.
- Multiple acceptance criteria in `alns.accept`.
  These include standard ones like hill-climbing and simulated annealing, and several variants of record-to-record travel and the great deluge criteria [@Dueck:1993].
- Several operator selection schemes in `alns.select`.
  These include the original (segmented) roulette wheel mechanism of @Ropke_Pisinger:2006, and an upper confidence bound bandit algorithm adapted from @Hendel:2022.
- Various stopping criteria in `alns.stop` based on maximum run-times or iterations.
  This includes a criterion that stops after a fixed number of iterations without improvement, which could be used to restart the search.
- Diagnostic statistics collection and plotting methods that can be accessed after solving.

The package can easily be installed through `pip`, and our detailed documentation is available [here](https://alns.readthedocs.io/).
To get started using `alns`, a user must provide:

- A solution state for your problem that implements an `objective()` function.
- An initial solution using this solution state.
- One or more destroy and repair operators tailored to your problem.

We provide a [quickstart template](https://alns.readthedocs.io/en/latest/setup/template.html) in our documentation, where these elements are listed as 'TODO'.
The documentation also provides several complete implementations of ALNS metaheuristics solving instances of the travelling salesman problem, capacitated vehicle routing problem, cutting stock problem, permutation flow shop problem, and the resource-constrained project scheduling problem.
These implementations will help users quickly get started solving their own problems using `alns`.

# References
