"""Classes for encoding molecules as phys-chem vector."""
from __future__ import annotations
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import Descriptors

from molpipeline.abstract_pipeline_elements.mol2any.mol2floatvector import (
    MolToDescriptorPipelineElement
)

RDKIT_DESCRIPTOR_DICT: dict[str, Callable[[Chem.Mol], float]]
RDKIT_DESCRIPTOR_DICT = dict(Descriptors.descList)

# MolWt is removed as ExactMolWt is already included.
# Ipc is removed because that's what we do. But for real: I have no clue, we need to ask Jenny.
DEFAULT_DESCRIPTORS = [name for name in RDKIT_DESCRIPTOR_DICT if name not in ['MolWt', 'Ipc']]


class MolToRDKitPhysChem(MolToDescriptorPipelineElement):
    _descriptor_list: list[str]

    def __init__(self,
                 descriptor_list: Optional[list[str]] = None,
                 name: str = "Mol2RDKitPhysChem",
                 n_jobs: int = 1
                 ) -> None:
        super().__init__(name=name, n_jobs=n_jobs)

        self._descriptor_list = descriptor_list or DEFAULT_DESCRIPTORS

    def _transform_single(self, value: Chem.Mol) -> npt.NDArray[np.float_]:
        return np.array([RDKIT_DESCRIPTOR_DICT[name](value) for name in self._descriptor_list])
