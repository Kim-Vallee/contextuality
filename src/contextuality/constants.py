# -*- coding: utf-8 -*-
#
# Written by Kim Vallée, https://github.com/Kim-Vallee.
#
# Created at 23/03/2022
#
# This code is licensed under the GNU GPLv3 license. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at https://www.gnu.org/licenses/gpl-3.0.fr.html#license-text.

import numpy as np

# ---------------------------
# Well known empirical models
# ---------------------------

EMPIRICAL_MODELS = {
    "CHSH": np.array([
        1 / 2, 0, 0, 1 / 2,
        3 / 8, 1 / 8, 1 / 8, 3 / 8,
        3 / 8, 1 / 8, 1 / 8, 3 / 8,
        1 / 8, 3 / 8, 3 / 8, 1 / 8
    ]),
    "PRBOX": np.array([
        0.5, 0., 0., 0.5,
        0.5, 0., 0., 0.5,
        0.5, 0., 0., 0.5,
        0., 0.5, 0.5, 0.
    ]),
    "MS": np.array([
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        0., 1., 0., 0.
    ]),
    "FD": np.array([
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        1., 0., 0., 0.
    ]),
}

pauli = {
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]])
}
