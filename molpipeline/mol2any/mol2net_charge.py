"""MolToNetCharge pipeline element."""

from __future__ import annotations

try:
    from typing import Any, Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.abstract_pipeline_elements.mol2any.mol2floatvector import (
    MolToDescriptorPipelineElement,
)
from molpipeline.utils.molpipeline_types import AnyTransformer, RDKitMol


class MolToNetCharge(MolToDescriptorPipelineElement):
    """PipelineElement for calculating molecules' net charge."""

    def __init__(
        self,
        standardizer: AnyTransformer | None = StandardScaler(),
        name: str = "MolToNetCharge",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize MolToNetCharge.

        Parameters
        ----------
        standardizer: bool
        name: str
        n_jobs: int
        """
        self._descriptor_list = ["NetCharge"]
        super().__init__(
            standardizer=standardizer,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )

    # pylint: disable=duplicate-code
    @property
    def n_features(self) -> int:
        """Return the number of features."""
        return len(self._descriptor_list)

    @property
    def descriptor_list(self) -> list[str]:
        """Return a copy of the descriptor list."""
        return self._descriptor_list[:]

    def pretransform_single(
        self, value: RDKitMol
    ) -> npt.NDArray[np.float_] | InvalidInstance:
        """Transform a single molecule to it's net charge.

        Based on https://github.com/rdkit/rdkit/discussions/4331

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule to transform.

        Returns
        -------
        Optional[npt.NDArray[np.float_]]
            Net charge of the given molecule.
        """
        # copy molecule since ComputeGasteigerCharges modifies the molecule inplace
        value_copy = Chem.Mol(value)
        Chem.rdPartialCharges.ComputeGasteigerCharges(value_copy)
        atoms_contributions = np.array(
            [atom.GetDoubleProp("_GasteigerCharge") for atom in value_copy.GetAtoms()]
        )
        if np.any(np.isnan(atoms_contributions)):
            return InvalidInstance(self.uuid, "NaN in Garsteiger charges", self.name)
        # sum up the charges and round to the nearest integer.
        net_charge = np.round(np.sum(atoms_contributions))
        return net_charge

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
        return parent_dict

    def set_params(self, **parameters: dict[str, Any]) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: dict[str, Any]
            Parameters to set

        Returns
        -------
        Self
            Self
        """
        super().set_params(**parameters)
        return self
