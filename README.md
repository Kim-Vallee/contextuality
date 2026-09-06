# Contextuality package

[![Software License][ico-license]](LICENSE)
![Version](https://img.shields.io/pypi/v/contextuality?color=brightgreen)
![Tests](https://github.com/Kim-Vallee/contextuality/actions/workflows/test.yml/badge.svg)

Contextuality is an open-source Python package for studying contextuality in measurement scenarios using the sheaf-theoretic approach to contextuality by [Abramsky and Brandenburger](http://arxiv.org/abs/1401.2561).

This package features:
- Custom measurement scenarios definitions.
- Predefined CHSH, KCBS, and Peres-Mermin measurement scenarios.
- Empirical models from probability vectors or matrices.
- Many features for empirical models, including generations, validations and other utilities.
- Build quantum empirical models from density matrices and PVMs.
- Compute: the Contextual Fraction (CF), the Signalling Fraction (SF), Dual contextual-fraction and more.
- Generate non-contextual and signalling polytopes in V- and H-representations (non-optimized).
- Combine empirical models using scalar multiplication, division, and convex mixtures.

## Example usage

The following example covers the main workflow: define a measurement scenario,
construct empirical models, inspect their probabilities, and compute their
contextual and signalling fractions.

```python
import numpy as np
from contextuality import EmpiricalModel, MeasurementScenario, MeasurementScenarioImplementations

# Define a scenario directly.
X = [0, 1, 2, 3, 4]
M = [[i, (i + 1) % 5] for i in X]
O = [0, 1]
custom_kcbs = MeasurementScenario(X, M, O)

# Or use one of the predefined scenarios.
chsh = MeasurementScenarioImplementations.chsh()
kcbs = MeasurementScenarioImplementations.kcbs()
peres_mermin = MeasurementScenarioImplementations.peres_mermin()

# Build a deterministic empirical model from one outcome position per context.
deterministic = EmpiricalModel(
    chsh,
    chsh.generate_deterministic([0, 0, 1, 2]),
)
print(deterministic.is_valid, deterministic.is_deterministic)

# A PR-box model is a useful contextual, no-signalling example.
pr_box = EmpiricalModel(
    chsh,
    np.array([
        [0.5, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.5],
        [0.0, 0.5, 0.5, 0.0],
    ]),
)
print(pr_box.probability_outcome(1, [0, 2], 0))
print(pr_box.maximum_incompatibility_of_marginals())
print(pr_box.compute_cf(solver="highs"))
print(pr_box.compute_sf(solver="highs"))

# Quantum realizations can be supplied as a density matrix and PVMs.
from qutip import basis, identity, ket2dm, sigmax, sigmaz, tensor

zero, one = basis(2, 0), basis(2, 1)
psi = (tensor(zero, one) - tensor(one, zero)) / np.sqrt(2)
rho = ket2dm(psi).unit().full()

sz, sx = sigmaz(), sigmax()
A0 = [ket2dm(state) for state in sz.eigenstates()[1]]
A1 = [ket2dm(state) for state in sx.eigenstates()[1]]
B0 = [ket2dm(state) for state in (-(sx + sz) / np.sqrt(2)).eigenstates()[1]]
B1 = [ket2dm(state) for state in ((sx - sz) / np.sqrt(2)).eigenstates()[1]]

pvms = [
    [tensor(projector, identity(2)).full() for projector in A0],
    [tensor(projector, identity(2)).full() for projector in A1],
    [tensor(identity(2), projector).full() for projector in B0],
    [tensor(identity(2), projector).full() for projector in B1],
]

quantum_model = EmpiricalModel(chsh)
quantum_model.quantum_realisation(rho, pvms)
print(quantum_model.compute_cf(solver="highs")["CF"])
```

More examples in the form of notebooks can be found in the [notebooks](notebooks) folder.

## Install

The package is working with pycddlib which is a python library for the double description method and you need to install cdd for it to work.
The installation procedure is on [their website](https://pycddlib.readthedocs.io/en/stable/quickstart.html#installing-cddlib-and-gmp). 
For example, for the aptitude package manager this amounts to:
```shell
$ sudo apt update
$ sudo apt install libcdd-dev libgmp-dev python3-dev
```

The package also uses solvers for linear programs and you need to install one.
The default is Mosek (see [installation instructions](https://www.mosek.com/downloads/)),
for which you can have a licence for free if you work in academia [here](https://www.mosek.com/products/academic-licenses/).
Another option is to go for [HiGHS solver](https://ergo-code.github.io/HiGHS/dev/interfaces/python/), which is free.

You can then install the package from pypi:
```shell
$ python -m pip install contextuality
```

## Documentation

The documentation is available online on [readthedocs](https://contextuality.readthedocs.io/en/latest/).

## Development

### Install from source

You can install the package directly from source:
```shell
$ git clone https://github.com/Kim-Vallee/contextuality.git
$ cd contextuality
$ poetry install --with dev
$ pip install -e . # or for poetry: poetry add --editable .
```

### Running tests

The tests are managed with pytest, which you can directly run with
```shell
$ pytest
```

### Building documentation

The documentation can be compiled in the [docs](docs) directory.

```bash
$ cd docs
$ make html
```

then navigate to [docs/build/html](docs/build/html) and open [index.html](docs/build/html/index.html) to access the documentation.


## Credits

- [Kim Vallée](https://github.com/Kim-Vallee) — Author and main contributor
- [Adel Sohbi](https://github.com/adelshb) — Author

## License

Contextuality is free and open-source software released under the GNU General Public Licence v3.0.

Please see [License File](LICENSE) for more information.

[ico-license]: https://img.shields.io/badge/License-GPLv3-blue.svg