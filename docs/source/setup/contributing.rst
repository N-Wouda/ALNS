Contributing
============

Conversations about development and issues take place in the `GitHub repository <https://github.com/N-Wouda/ALNS/>`_.
Feel free to open a new issue if you have something to discuss.


Setting up a local installation
-------------------------------

First, fork the ``alns`` repository from the `GitHub website <https://github.com/N-Wouda/ALNS/fork>`_.
Then, clone your new fork to some local environment:

.. code-block:: shell

   git clone https://github.com/<your username>/ALNS.git

Now, change into the ALNS directory, and set-up the virtual environment using ``poetry``:

.. code-block:: shell

   cd ALNS

   pip install --upgrade poetry
   poetry install --with examples --all-extras

This might take a few minutes, but only needs to be done once.
Now make sure everything runs smoothly, by executing the test suite:

.. code-block:: shell

   poetry run pytest

.. note::

   If you use `pre-commit <https://pre-commit.com/>`_, you can use our pre-commit configuration file to set that up too.
   Simply type:

   .. code-block:: shell

      pre-commit install

   After this completes, style and typing issues are automatically checked whenever you make a new commit to your feature branch.


Committing changes
------------------

We use pull requests to develop the ``alns`` package.
For a pull request to be accepted, you must meet the below requirements.
This greatly reduces the job of maintaining and releasing the software.

- **One branch. One feature.**
  Branches are cheap and GitHub makes it easy to merge and delete branches with a few clicks.
  Avoid the temptation to lump in a bunch of unrelated changes when working on a feature, if possible.
  This helps us keep track of what has changed when preparing a release.
- Commit messages should be clear and concise.
  This means a subject line of less than 80 characters, and, if necessary, a blank line followed by a commit message body.
- Code submissions should always include tests.
- Each function, class, method, and attribute needs to be documented using docstrings.
  We conform to the `NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.
- If you are adding new functionality, you need to add it to the documentation by editing (or creating) the appropriate file in ``docs/source/``.
- Make sure your documentation changes parse correctly.
  See the documentation in the ``docs/`` directory for details on how to build the documentation locally.
