"""Classes for encoding molecules as phys-chem vector."""
# pylint: disable=too-many-arguments

from __future__ import annotations
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import Descriptors

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
        mean: Optional[npt.NDArray[np.float_]] = None,
        std: Optional[npt.NDArray[np.float_]] = None,
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
            normalize=normalize, name=name, n_jobs=n_jobs, mean=mean, std=std
        )

    @property
    def n_features(self) -> int:
        """Return the number of features."""
        return len(self._descriptor_list)

    @property
    def descriptor_list(self) -> list[str]:
        """Return a copy of the descriptor list."""
        return self._descriptor_list[:]

    @property
    def params(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        return super().params

    def copy(self) -> MolToRDKitPhysChem:
        """Create a copy of the object."""
        return MolToRDKitPhysChem(**self.params)

    def _transform_single(self, value: Chem.Mol) -> npt.NDArray[np.float_]:
        return np.array(
            [RDKIT_DESCRIPTOR_DICT[name](value) for name in self._descriptor_list]
        )
