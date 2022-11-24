Installation instructions
=========================

The most straightforward way to use the ``alns`` package in your project is to install via *pip*, like so:

.. code-block:: shell

   pip install alns


Installing from source
----------------------

To install the latest version of ``alns`` directly from the GitHub repository, you can use *pip*, like so:

.. code-block:: shell

   pip install 'alns @ git+https://github.com/N-Wouda/alns'

This can be useful to get updates that have not yet made it to the Python package index.


.. _running-locally:

Running the examples locally
----------------------------

To run the example notebooks locally, first clone the repository:

.. code-block::

   git clone https://github.com/N-Wouda/ALNS.git

Then, make sure your Python version has ``poetry``:

.. code-block::

   pip install --upgrade poetry

Now, go into the ALNS repository and set-up a virtual environment.
We want a virtual environment that also contains all dependencies needed to run the example notebooks, so we also need to install the optional ``examples`` dependency group.
This goes like so:

.. code-block::

   cd ALNS
   poetry install --with examples

This might take a few minutes to resolve, but only needs to be done once.
After setting everything up, simply open the jupyter notebooks:

.. code-block::

   poetry run jupyter notebook

This will open up a page in your browser, where you can navigate to the example notebooks in the ``examples/`` folder!
