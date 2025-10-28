.. CF documentation master file, created by
   sphinx-quickstart on Wed Mar 23 17:05:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting started
===============

Installation
------------

The package is available on PyPi so you can install it directly from pip:

.. code-block:: console

      $ pip install contextuality

Contributing
------------

If you want to contribute, feel free to make pull requests `on the github page <https://github.com/Kim-Vallee/contextuality>`_.

If you want to install from source you can clone from github:

.. code-block:: console

      $ git clone https://github.com/Kim-Vallee/contextuality.git
      $ cd contextuality
      $ poetry install --with dev
      $ pytest # Checking that the tests are working

Code example
------------

.. code-block:: python

      from contextuality.measurement_scenario import MeasurementScenario, MeasurementScenarioImplementations
      import numpy as np
      from contextuality.empirical_model import EmpiricalModel
      from contextuality.utils import compute_max_CF, compute_deterministic_fraction, compute_signaling_fraction, compute_NCF

      # Defining the contextuality scenario
      X = [0, 1, 2, 3, 4]
      M = [[i, (i + 1) % 5] for i in X]
      O = [0, 1]
      kcbs = MeasurementScenario(X, M, O)

      # Equivalently from pre-defined scenarios
      chsh = MeasurementScenarioImplementations.CHSH()

      # We can make a simple empirical model...
      empirical_model_ex = EmpiricalModel(kcbs, np.array([1,0,0,0]*5))

      # ... or make a quantum realization of an empirical model
      empirical_model = EmpiricalModel(kcbs)
      meas = np.zeros((5, 2, 3, 3))  # shape = number mesurements, number of outcomes, dimension of state (d x d)
      N = 1 / np.sqrt(1 + np.cos(np.pi / 5))
      for i in range(5):
        vec = N * np.array([np.cos(4 * np.pi * i / 5), np.sin(4 * np.pi * i / 5), np.sqrt(np.cos(np.pi / 5))])
        meas[i][1] = np.outer(vec, vec)
        meas[i][0] = np.eye(3) - meas[i][1]

      psi = np.array([0, 0, 1])
      rho = np.outer(psi, psi)
      empirical_model.quantum_realisation(rho, meas)

      # We can compute the contextual fraction
      ncf_empirical_model = empirical_model.compute_NCF(solver="MOSEK")["NCF"]

      # The signalling fraction from the utils
      sf_empirical_model = compute_signaling_fraction(empirical_model)["SF"]

      # Then there are plenty of functions to use from utils
      result = compute_max_CF(kcbs, eta=0.3, sigma=0.5) # Experimental

      print(result['EmpiricalModel'].vector)

      df = compute_deterministic_fraction(result["EmpiricalModel"], verbose=False)
      print(df)

      CF_result = compute_NCF(result['EmpiricalModel'], verbose=False)

Documentation structure
-----------------------


.. toctree::
   :maxdepth: 2

   modules/empirical_model
   modules/measurement_scenario
   modules/utils
   modules/printing



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
