"""Classes for encoding molecules as phys-chem vector."""

# pylint: disable=too-many-arguments

from __future__ import annotations

from typing import Any, Callable, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.abstract_pipeline_elements.mol2any.mol2floatvector import (
    MolToDescriptorPipelineElement,
)
from molpipeline.utils.molpipeline_types import AnyTransformer, RDKitMol

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
        standardizer: Optional[AnyTransformer] = StandardScaler(),
        name: str = "Mol2RDKitPhysChem",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToRDKitPhysChem.

        Parameters
        ----------
        descriptor_list: Optional[list[str]], optional (default=None)
            List of descriptor names to calculate. If None, DEFAULT_DESCRIPTORS are used.
        standardizer: Optional[AnyTransformer], optional (default=StandardScaler())
            Standardizer to use.
        name: str, optional (default="Mol2RDKitPhysChem")
            Name of the PipelineElement.
        n_jobs: int, optional (default=1)
            Number of jobs to use for parallelization.
        uuid: Optional[str], optional (default=None)
            UUID of the PipelineElement. If None, a new UUID is generated.
        """
        self._descriptor_list = descriptor_list or DEFAULT_DESCRIPTORS
        super().__init__(
            standardizer=standardizer,
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
        """Return a copy of the descriptor list."""
        return self._descriptor_list[:]

    def pretransform_single(
        self, value: RDKitMol
    ) -> Union[npt.NDArray[np.float_], InvalidInstance]:
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
            return InvalidInstance(self.uuid, "NaN in descriptor vector", self.name)
        return vec

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
            parent_dict["descriptor_list"] = copy.deepcopy(self._descriptor_list)
        else:
            parent_dict["descriptor_list"] = self._descriptor_list
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
        parameters_shallow_copy = dict(parameters)
        descriptor_list = parameters_shallow_copy.pop("descriptor_list", None)
        if descriptor_list is not None:
            self._descriptor_list = descriptor_list  # type: ignore
        super().set_params(**parameters_shallow_copy)
        return self
