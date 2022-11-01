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

# Introduction

The adaptive large neighbourhood search (ALNS) metaheuristic algorithm, introduced by TODO, has quickly become one of the most used metaheuristics in the field of combinatorial optimisation TODO.
At its core, ALNS implements an iterative ruin-and-recreate cycle that is applied until some stopping criterion is met.
The algorithm starts with some initial solution.
In each iteration, the current solution is destroyed using some operator and repaired using another to obtain a candidate solution.
This candidate solution is then evaluated by an acceptance criterion, and possibly replaces the current solution.
Based on the outcome of that evaluation, the likelihood that some operators are selected is modified - this is the adaptive part of the metaheuristic.

The `alns` package provides a library containing a complete implementation of ALNS as described in @Pisinger_Ropke:2010.
It additionally includes multiple stopping criteria, a large set of acceptance criteria based on @Santini_et_al:2018, and multiple operator selection schemes. 
Furthermore, it supports many other single-trajectory neighbourhood search algorithms as special cases, including iterated local search (ILS), variable neighbourhood search (VND), and greedy randomized adaptive search (GRASP).
The package can thus be used for research into heuristic algorithms for various problems in industry, including TODO, TODO, and TODO.
Additionally, it can be used for research into improving heuristic algorithms themselves, for example by replacing TODO with a machine learning methods TODO.

# Features

The `alns` library is a Python package that offers:
- TODO
- TODO
- TODO

# Statement of need

TODO

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

TODO

# References
