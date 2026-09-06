# Contributing to Contextuality

First and foremost, thank you for considering a contribution!

Contributions are welcome, including bug fixes, new features, improvements to existing functionality, documentation, tests, and examples.

## Getting started

Fork the repository and clone your fork:

```bash
git clone https://github.com/<your-username>/contextuality.git
cd contextuality
```

Install the development dependencies with Poetry:

```bash
poetry install
```

Run the test suite:

```bash
poetry run pytest
```

## Making changes

Create a new branch for your changes:

```bash
git checkout -b feature/my-feature
```

Please keep changes focused and avoid unrelated modifications.

For new functionality, add or update tests and documentation systematically.
For mathematical or scientific changes, please include relevant references or examples when useful.

## Pull requests

Push your branch to your fork:

```bash
git push origin feature/my-feature
```

Then open a pull request against the `main` branch.

Please make sure that:

* tests pass locally;
* new functionality is tested;
* documentation is updated;
* the pull request describes the changes (following the template).

All pull requests will be reviewed before being merged into `main`.

## Reporting issues

Please use the GitHub issue templates to report bugs, request features, or suggest documentation improvements.

For bugs, include a minimal example and the relevant Python and Contextuality versions whenever possible.
