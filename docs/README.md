# Documentation

This directory hosts the documentation. 
We use Sphinx for this.
The documentation has a few unique dependencies that are listed in the optional `docs` group in the top-level `pyproject.toml`.
If you want to build the documentation, make sure to install that group (using `poetry install --with docs` or `--only docs`).

The Makefile in this directory can be used to build the documentation.
Running the command `poetry run make help` from this directory provides an overview of the available options.
In particular `poetry run make html` is useful, as that will build the documentation in the exact same way as it will be displayed on readthedocs later.

> Alternatively, one can run `poetry run make html --directory=docs` from the project root as well.

Finally, all Sphinx-related settings are configured in `docs/source/conf.py`.

## Skipping the example notebooks

The example notebooks are all executed by default when building the documentation.
This can take a long time, so it is possible to skip execution when building the documentation locally.
To do so, simply set the `SKIP_NOTEBOOKS` environmental variable when building the documentation, like so
```shell
SKIP_NOTEBOOKS=1 poetry run make html --directory=docs
```
