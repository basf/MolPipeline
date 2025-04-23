"""MolToNetCharge pipeline element."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.abstract_pipeline_elements.mol2any.mol2floatvector import (
    MolToDescriptorPipelineElement,
)

if TYPE_CHECKING:
    from molpipeline.utils.molpipeline_types import RDKitMol

# Methods to compute the net charge of a molecule.
# - "formal_charge" uses the formal charges of the atoms
#   - uses rdkit.Chem.rdmolops.GetFormalCharge
# - "gasteiger" uses the Gasteiger charges of the atoms with
#   - uses rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges
MolToNetChargeMethod: TypeAlias = Literal["formal_charge", "gasteiger"]


class MolToNetCharge(MolToDescriptorPipelineElement):
    """PipelineElement for calculating molecules' net charge."""

    def __init__(
        self,
        charge_method: MolToNetChargeMethod = "formal_charge",
        name: str = "MolToNetCharge",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize MolToNetCharge.

        Parameters
        ----------
        charge_method: MolToNetChargeMethod, optional (default="formal_charge")
            Policy how to compute the net charge of a molecule.
            "formal_charge" uses sum of the formal charges assigned to each atom.
            "gasteiger" computes the Gasteiger partial charges and returns the rounded
            sum over the atoms.
        name: str, optional
            Name of the pipeline element, by default "MolToNetCharge"
        n_jobs: int, optional
            Number of jobs to run in parallel, by default 1
        uuid: str, optional
            UUID of the pipeline element, by default None

        """
        self._descriptor_list = ["NetCharge"]
        self._feature_names = self._descriptor_list
        self._charge_method = charge_method
        # pylint: disable=R0801
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )

    @property
    def n_features(self) -> int:
        """Return the number of features."""
        return len(self._descriptor_list)

    @property
    def descriptor_list(self) -> list[str]:
        """Return a copy of the descriptor list. Alias of `feature_names`."""
        return self._descriptor_list[:]

    def _get_net_charge_gasteiger(
        self,
        value: RDKitMol,
    ) -> npt.NDArray[np.float64] | InvalidInstance:
        """Transform a single molecule to it's net charge using Gasteiger charges.

        Based on https://github.com/rdkit/rdkit/discussions/4331

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule to transform.

        Returns
        -------
        Optional[npt.NDArray[np.float64]]
            Net charge of the given molecule.

        """
        # copy molecule since ComputeGasteigerCharges modifies the molecule inplace
        value_copy = Chem.Mol(value)
        rdPartialCharges.ComputeGasteigerCharges(value_copy)
        atoms_contributions = np.array(
            [atom.GetDoubleProp("_GasteigerCharge") for atom in value_copy.GetAtoms()],
        )
        if np.any(np.isnan(atoms_contributions)):
            return InvalidInstance(self.uuid, "NaN in Gasteiger charges", self.name)
        # sum up the charges and round to the nearest integer.
        return np.round(np.sum(atoms_contributions, keepdims=True))

    def pretransform_single(
        self,
        value: RDKitMol,
    ) -> npt.NDArray[np.float64] | InvalidInstance:
        """Transform a single molecule to it's net charge.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule to transform.

        Raises
        ------
        ValueError
            If the charge method is not supported.

        Returns
        -------
        npt.NDArray[np.float64] | InvalidInstance
            Net charge of the given molecule.

        """
        if self._charge_method == "formal_charge":
            return np.array([Chem.GetFormalCharge(value)], dtype=np.float64)
        if self._charge_method == "gasteiger":
            return self._get_net_charge_gasteiger(value)
        raise ValueError(f"Unknown charge policy: {self._charge_method}")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the parameters of the pipeline element.

        Parameters
        ----------
        deep: bool
            If true create a deep copy of the parameters

        Returns
        -------
        dict[str, Any]
            Parameter of the pipeline element.

        """
        parent_dict = dict(super().get_params(deep=deep))
        if deep:
            parent_dict["charge_policy"] = copy.deepcopy(self._charge_method)
        else:
            parent_dict["charge_policy"] = self._charge_method
        return parent_dict

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: Any
            Parameters to set

        Returns
        -------
        Self
            Self

        """
        parameters_shallow_copy = dict(parameters)
        charge_policy = parameters_shallow_copy.pop("charge_policy", None)
        if charge_policy is not None:
            self._charge_method = charge_policy
        super().set_params(**parameters_shallow_copy)
        return self
