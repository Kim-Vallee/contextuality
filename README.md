# Contextuality package

[![Software License][ico-license]](LICENSE)
![Version][ico-version]

This project is a starter project to have many tools to compute various quantities in measurement scenarios as defined
by [Abramsky and Brandenburger](http://arxiv.org/abs/1401.2561). It can be used in a variety of cases.

## Install

To install the package from pypi you can simply run pip:
```shell
$ python -m pip install contextuality
```

### Developers

If you wish to improve the package you can install from the sources with poetry:
```shell
$ poetry install --with dev
```

## Documentation

The documentation is available on [readthedocs](https://contextuality.readthedocs.io/en/latest/).

### Compile documentations

The documentation can be compiled in the [docs](docs) directory.

```bash
$ cd docs
$ make html
```

then navigate to [docs/build/html](docs/build/html) and open [index.html](docs/build/html/index.html) to access the
documentation.

## Usage example

```python
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
```

## Notebooks

Examples in the form of notebooks can be found in the [notebooks](notebooks) folder.

## Credits

- [Kim Vallée](https://github.com/Kim-Vallee) -- Author and main contributor
- [Adel Sohbi](https://github.com/adelshb) -- Author

## License

The CC BY-NC 4.0. Please see [License File](LICENSE) for more information.

[ico-version]: https://img.shields.io/badge/Version-1.0.3-brightgreen.svg?style=flat-square

[ico-license]: https://img.shields.io/badge/License-CC_BYNC_4.0-brightgreen.svg?style=flat-square