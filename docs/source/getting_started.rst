Getting started
===============

This project is a starter project to have many tools to compute various quantities in measurement scenarios as defined
by `Abramsky and Brandenburger <http://arxiv.org/abs/1401.2561>`_. It can be used in a variety of cases.

Install
-------

The package is working with pycddlib, thus you need to install cdd. See directly on `their website <https://pycddlib.readthedocs.io/en/stable/quickstart.html#installing-cddlib-and-gmp>`_. For aptitude this amounts to:

.. code-block:: shell

   $ sudo apt update
   $ sudo apt install libcdd-dev libgmp-dev python3-dev

The package also uses solvers for linear programs. 
The default is Mosek (see `installation instructions <https://www.mosek.com/downloads/>`_),
for which you can have a licence for free if you work in academia `here <https://www.mosek.com/products/academic-licenses/>`_.
Another option is to go for `HiGHS solver <https://ergo-code.github.io/HiGHS/dev/interfaces/python/>`_, which is free.

Then you can either install the package from pypi:

.. code-block:: shell

   $ python -m pip install contextuality

Or you can install it from source:

.. code-block:: shell

   $ git clone https://github.com/Kim-Vallee/contextuality.git
   $ cd contextuality
   $ poetry install
   $ poetry build
   $ python -m pip install dist/contextuality-<version>-py3-none-any.whl

Contribute
~~~~~~~~~~

If you wish to improve the package you can install from the sources with poetry and make pull requests:

.. code-block:: shell

   $ git clone https://github.com/Kim-Vallee/contextuality.git
   $ cd contextuality
   $ poetry install --with dev
   $ pip install -e . # or for poetry:
   $ poetry add --editable .

Documentation
-------------

The documentation is available on `readthedocs <https://contextuality.readthedocs.io/en/latest/>`_.

Compile documentations
~~~~~~~~~~~~~~~~~~~~~~

The documentation can be compiled in the ``docs`` directory.

.. code-block:: bash

   $ cd docs
   $ make html

then navigate to ``docs/build/html`` and open ``index.html`` to access the documentation.

Usage example
-------------

.. code-block:: python

   from contextuality.measurement_scenario import MeasurementScenario, MeasurementScenarioImplementations
   import numpy as np
   from contextuality.empirical_model import EmpiricalModel
   from contextuality.utils import compute_max_cf, compute_deterministic_fraction

   # Defining the contextuality scenario
   X = [0, 1, 2, 3, 4]
   M = [[i, (i + 1) % 5] for i in X]
   O = [0, 1]
   kcbs = MeasurementScenario(X, M, O)

   # Equivalently from pre-defined scenarios
   chsh = MeasurementScenarioImplementations.CHSH()

   # We can make a simple empirical model...
   empirical_model_ex = EmpiricalModel(kcbs, np.array([1, 0, 0, 0] * 5))

   # ... or make a quantum realization of an empirical model
   empirical_model = EmpiricalModel(kcbs)
   meas = np.zeros((5, 2, 3, 3))  # shape = number measurements, number of outcomes, dimension of state (d x d)
   N = 1 / np.sqrt(1 + np.cos(np.pi / 5))
   for i in range(5):
       vec = N * np.array([np.cos(4 * np.pi * i / 5), np.sin(4 * np.pi * i / 5), np.sqrt(np.cos(np.pi / 5))])
       meas[i][1] = np.outer(vec, vec)
       meas[i][0] = np.eye(3) - meas[i][1]

   psi = np.array([0, 0, 1])
   rho = np.outer(psi, psi)
   empirical_model.quantum_realisation(rho, meas)

   # We can compute the contextual fraction
   ncf_empirical_model = empirical_model.compute_cf(solver="MOSEK")["NCF"]

   # The signalling fraction from the utils
   sf_empirical_model = empirical_model.compute_sf()['SF']

   # Then there are plenty of functions to use from utils
   result = compute_max_cf(kcbs, eta=0.3, sigma=0.5)  # Experimental

   print(result['EmpiricalModel'].vector)

   df = compute_deterministic_fraction(result["EmpiricalModel"], verbose=False)
   print(df)

   CF_result = result['EmpiricalModel'].compute_cf()

Notebooks
---------

Examples in the form of notebooks can be found in the ``notebooks`` folder.

Credits
-------

- `Kim Vallée <https://github.com/Kim-Vallee>`_ — Author and main contributor
- `Adel Sohbi <https://github.com/adelshb>`_ — Author

License
-------

This work is licensed under GNU GPLv3. Please see the `License File <LICENSE>`_ for more information.