Getting help
============

Conversations about development and issues take place in the `GitHub repository <https://github.com/N-Wouda/ALNS/>`_.
Feel free to open a new issue if you have something to discuss.

Submitting a bug report
-----------------------

Open a new issue in the repository, using the "Bug report" template.
To limit the amount of time needed to triage your problem, please do the following:

- Include a short, self-contained code snippet that reproduces the problem.
- Specify the version information of the ``alns`` installation you use.
  You can do this by including the output of :meth:`~alns.show_versions.show_versions`:

  .. code-block:: python

     import alns
     alns.show_versions()

  in your report.
  You can run this from the command line as

  .. code-block:: shell

     python -c 'import alns; alns.show_versions()'
