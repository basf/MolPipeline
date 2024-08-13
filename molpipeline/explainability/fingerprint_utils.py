"""Utility functions for explainability."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import numpy.typing as npt

from molpipeline.abstract_pipeline_elements.core import RDKitMol
from molpipeline.mol2any import MolToMorganFP
from molpipeline.utils.substructure_handling import AtomEnvironment


def assign_prediction_importance(
    bit_dict: dict[int, Sequence[AtomEnvironment]], weights: npt.NDArray[np.float64]
) -> dict[int, float]:
    """Assign the prediction importance.

    Originally from Christian W. Feldmann
    https://github.com/c-feldmann/compchemkit/blob/64e5543e2b8f72e93711186b2e0b42366820fb52/compchemkit/molecular_heatmaps.py#L28

    Parameters
    ----------
    bit_dict : dict[int, Sequence[AtomEnvironment]]
        The bit dictionary.
    weights : npt.NDArray[np.float64]
        The weights.

    Returns
    -------
    dict[int, float]
        The atom contribution.
    """
    atom_contribution: dict[int, float] = defaultdict(lambda: 0)
    for bit, atom_env_list in bit_dict.items():  # type: int, Sequence[AtomEnvironment]
        n_machtes = len(atom_env_list)
        for atom_set in atom_env_list:
            for atom in atom_set.environment_atoms:
                atom_contribution[atom] += weights[bit] / (
                    len(atom_set.environment_atoms) * n_machtes
                )
    if not np.isclose(sum(weights), sum(atom_contribution.values())).all():
        raise AssertionError(
            f"Weights and atom contributions don't sum to the same value:"
            f" {weights.sum()} != {sum(atom_contribution.values())}"
        )
    return atom_contribution


def fingerprint_shap_to_atomweights(
    mol: RDKitMol, fingerprint_element: MolToMorganFP, shap_mat: npt.NDArray[np.float64]
) -> list[float]:
    """Convert SHAP values to atom weights.

    Originally from Christian W. Feldmann
    https://github.com/c-feldmann/compchemkit/blob/64e5543e2b8f72e93711186b2e0b42366820fb52/compchemkit/molecular_heatmaps.py#L15

    Parameters
    ----------
    mol : RDKitMol
        The molecule.
    fingerprint_element : MolToMorganFP
        The fingerprint element.
    shap_mat : npt.NDArray[np.float64]
        The SHAP values.

    Returns
    -------
    list[float]
        The atom weights.
    """
    bit_atom_env_dict: dict[int, Sequence[AtomEnvironment]]
    bit_atom_env_dict = dict(
        fingerprint_element.bit2atom_mapping(mol)
    )  # MyPy invariants make me do this.
    atom_weight_dict = assign_prediction_importance(bit_atom_env_dict, shap_mat)
    atom_weight_list = [
        atom_weight_dict[a_idx] if a_idx in atom_weight_dict else 0
        for a_idx in range(mol.GetNumAtoms())
    ]
    return atom_weight_list
