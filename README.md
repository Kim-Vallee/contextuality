# CF

[![Software License][ico-license]](LICENSE)
![Version][ico-version]

This project aims at having tools to compute the contextual fraction, and various associated quantities.
It has many linear programs in it, that could be used for other applications.

## Install

Via pip

``` bash
$ python -m pip install .
```

or dynamically

``` bash
$ python -m pip install -e .
```

## Usage

```python
from CF.measurement_scenario import MeasurementScenario
import numpy as np
from CF.empirical_model import EmpiricalModel
from CF.utils import compute_max_CF, compute_deterministic_fraction, compute_signaling_fraction, compute_NCF

X = [i for i in range(4)]
M = [[0, 2], [0, 3], [1, 2], [1, 3]]
O = [0, 1]
kcbs = MeasurementScenario(X, M, O)

meas = np.zeros((5, 2, 3, 3))  # shape = number mesurements, number of outcomes, dimension of state (d x d)
N = 1 / np.sqrt(1 + np.cos(np.pi / 5))
for i in range(5):
    vec = N * np.array([np.cos(4 * np.pi * i / 5), np.sin(4 * np.pi * i / 5), np.sqrt(np.cos(np.pi / 5))])
    meas[i][1] = np.outer(vec, vec)
    meas[i][0] = np.eye(3) - meas[i][1]

psi = np.array([0, 0, 1])
rho = np.outer(psi, psi)

empirical_model = EmpiricalModel(kcbs)
empirical_model.quantum_realisation(rho, meas)

result = compute_max_CF(kcbs, eta=0.3, sigma=0.5)

# result = LP_inequality(chsh)
# ve = EmpiricalModel(chsh, result['p'])
print(result['EmpiricalModel'].vector)

df = compute_deterministic_fraction(result["EmpiricalModel"], verbose=False)
print(df)
sf = compute_signaling_fraction(result["EmpiricalModel"], verbose=False)
print(sf)

CF_result = compute_NCF(result['EmpiricalModel'], verbose=False)

print(CF_result)
```

## Credits

- [Kim Vallée](https://github.com/Kim-Vallee)
- [Adel Sohbi](https://github.com/adelshb)

## License

The Apache 2.0 License. Please see [License File](LICENSE) for more information.

[ico-version]: https://img.shields.io/badge/version-1.0-brightgreen.svg?style=flat-square
[ico-license]: https://img.shields.io/badge/license-Apache-brightgreen.svg?style=flat-square