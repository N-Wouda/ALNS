# Contributing guidelines

This page explains how you can contribute to the development of `alns` by submitting patches, extensions, or examples.

The `alns` library is developed on [GitHub](https://github.com/N-Wouda/ALNS) using the [Git](https://git-scm.com/) version control system.

## Where to go for discussions and help?

Conversations about development and issues take place in the GitHub repository.
Feel free to open a new issue if you have something to discuss.

## Submitting a bug report

- Include a short, self-contained code snippet that reproduces the problem
- Specify the version information of the `alns` installation you use. 
  You can do this by including the output of `import alns; alns.show_versions()` in your report.

## Making changes to the code

For a pull request to be accepted, you must meet the below requirements. 
This greatly helps in keeping the job of maintaining and releasing the software a shared effort.

- **One branch. One feature.** 
  Branches are cheap and GitHub makes it easy to merge and delete branches with a few clicks. 
  Avoid the temptation to lump in a bunch of unrelated changes when working on a feature, if possible. 
  This helps us keep track of what has changed when preparing a release.
- Commit messages should be clear and concise. 
  This means a subject line of less than 80 characters, and, if necessary, a blank line followed by a commit message body.
- Code submissions should always include tests.
- Each function, class, method, and attribute needs to be documented using docstrings.
  We conform to the `numpy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.
- If you are adding new functionality, you need to add it to the documentation by editing (or creating) the appropriate file in ``docs/source``.
- Make sure your documentation changes parse correctly. Change into the top-level `docs/` directory and type:
  ```shell
  make clean
  make html
  ```
  Check that the build output does not have *any* warnings due to your changes.
