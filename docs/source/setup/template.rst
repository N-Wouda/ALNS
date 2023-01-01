Quickstart template
===================

The ``alns`` library provides the :class:`~alns.ALNS` algorithm and various
acceptance criteria in :mod:`alns.accept`, operator selection schemes in
:mod:`alns.select`, and stopping criteria in :mod:`alns.stop`.

You should provide the following:

* A solution state for your problem that implements an ``objective()`` function.
* An initial solution.
* One or more destroy and repair operators tailored to your problem. Each destroy
  operator should copy the passed-in state; see
  :meth:`~alns.ALNS.ALNS.add_destroy_operator` for details.

.. note::

   The ``alns`` package assumes your problem is a minimisation problem. If you
   instead want to maximise some objective, you can use that

   .. math::

      \arg \max_x f(x) = \arg \min_x -f(x),

   that is, the solution :math:`x` that maximises :math:`f(x)` is the same
   :math:`x` that mimimises :math:`-f(x)`. In your ALNS implementation, you
   should thus implement your objective as :math:`-f(x)`.

The following is a quickstart template that can help you get started:

.. code-block:: python

    from alns import ALNS
    from alns.accept import HillClimbing
    from alns.select import RandomSelect
    from alns.stop import MaxRuntime

    import numpy.random as rnd


    class ProblemState:
        # TODO add attributes that encode a solution to the problem instance

        def objective(self) -> float:
            # TODO implement the objective function
            pass


    def initial_state() -> ProblemState:
        # TODO implement a function that returns an initial solution
        pass


    def destroy(current: ProblemState, rnd_state: rnd.RandomState) -> ProblemState:
        # TODO implement how to destroy the current state, and return the destroyed
        #  state. Make sure to (deep)copy the current state before modifying!
        pass


    def repair(destroyed: ProblemState, rnd_state: rnd.RandomState) -> ProblemState:
        # TODO implement how to repair a destroyed state, and return it
        pass


    # Create the initial solution
    init_sol = initial_state()
    print(f"Initial solution objective is {init_sol.objective()}.")

    # Create ALNS and add one or more destroy and repair operators
    alns = ALNS(rnd.RandomState(seed=42))
    alns.add_destroy_operator(destroy)
    alns.add_repair_operator(repair)

    # Configure ALNS
    select = RandomSelect(num_destroy=1, num_repair=1)  # see alns.select for others
    accept = HillClimbing()  # see alns.accept for others
    stop = MaxRuntime(60)  # 60 seconds; see alns.stop for others

    # Run the ALNS algorithm
    result = alns.iterate(init_sol, select, accept, stop)

    # Retrieve the final solution
    best = result.best_state
    print(f"Best heuristic solution objective is {best.objective()}.")

.. note::

    Have a look at the examples to get a feeling for how to implement the TODOs
    in the quickstart template!
