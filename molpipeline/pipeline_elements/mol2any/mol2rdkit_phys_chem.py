"""Classes for encoding molecules as phys-chem vector."""
# pylint: disable=too-many-arguments

from __future__ import annotations
from typing import Any, Callable, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy
import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]
from rdkit.Chem import Descriptors

from molpipeline.abstract_pipeline_elements.core import NoneHandlingOptions
from molpipeline.abstract_pipeline_elements.mol2any.mol2floatvector import (
    MolToDescriptorPipelineElement,
)

RDKIT_DESCRIPTOR_DICT: dict[str, Callable[[Chem.Mol], float]]
RDKIT_DESCRIPTOR_DICT = dict(Descriptors.descList)

# MolWt is removed as ExactMolWt is already included.
# Ipc is removed because it causes trouble with numpy.
DEFAULT_DESCRIPTORS = [
    name for name in RDKIT_DESCRIPTOR_DICT if name not in ["MolWt", "Ipc"]
]


class MolToRDKitPhysChem(MolToDescriptorPipelineElement):
    """PipelineElement for creating a Descriptor vector based on RDKit phys-chem properties."""

    _descriptor_list: list[str]

    def __init__(
        self,
        descriptor_list: Optional[list[str]] = None,
        normalize: bool = True,
        name: str = "Mol2RDKitPhysChem",
        n_jobs: int = 1,
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = None,
    ) -> None:
        """Initialize MolToRDKitPhysChem.

        Parameters
        ----------
        descriptor_list: Optional[list[str]]
        normalize: bool
        name: str
        n_jobs: int
        """
        self._descriptor_list = descriptor_list or DEFAULT_DESCRIPTORS
        super().__init__(
            normalize=normalize,
            name=name,
            n_jobs=n_jobs,
            none_handling=none_handling,
            fill_value=fill_value,
        )

    @property
    def n_features(self) -> int:
        """Return the number of features."""
        return len(self._descriptor_list)

    @property
    def descriptor_list(self) -> list[str]:
        """Return a copy of the descriptor list."""
        return self._descriptor_list[:]

    def _transform_single(self, value: RDKitMol) -> Optional[npt.NDArray[np.float_]]:
        """Transform a single molecule to a descriptor vector.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule to transform.

        Returns
        -------
        Optional[npt.NDArray[np.float_]]
            Descriptor vector for given molecule. None if calculation failed.
        """
        vec = np.array(
            [RDKIT_DESCRIPTOR_DICT[name](value) for name in self._descriptor_list]
        )
        if np.any(np.isnan(vec)):
            return None
        return vec

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters.

        Parameters
        ----------
        deep: bool
            If true create a deep copy of the parameters

        Returns
        -------
        dict[str, Any]
            Parameters
        """
        parent_dict = dict(super().get_params(deep=deep))
        if deep:
            parent_dict["descriptor_list"] = copy.deepcopy(self._descriptor_list)
        else:
            parent_dict["descriptor_list"] = self._descriptor_list
        return parent_dict

    def set_params(self, parameters: dict[str, Any]) -> Self:
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
        parameters_shallow_copy = dict(parameters)
        descriptor_list = parameters_shallow_copy.pop("descriptor_list", None)
        if descriptor_list is not None:
            self._descriptor_list = descriptor_list
        super().set_params(parameters_shallow_copy)
        return self
