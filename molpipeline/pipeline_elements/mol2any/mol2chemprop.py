"""Classes for encoding molecules as phys-chem vector."""
# pylint: disable=too-many-arguments

from __future__ import annotations
from typing import Any, Iterable, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy
import numpy as np
import numpy.typing as npt
from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]
from chemprop.v2 import data as cp_data

from rdkit import Chem


from molpipeline.abstract_pipeline_elements.core import NoneHandlingOptions
from molpipeline.abstract_pipeline_elements.mol2any.mol2floatvector import (
    MolToDescriptorPipelineElement,
)


class MolToChemprop(MolToDescriptorPipelineElement):
    """PipelineElement for creating a Descriptor vector based on RDKit phys-chem properties."""


    def __init__(
        self,
        normalize: bool = True,
        name: str = "Mol2Chemprop",
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
        super().__init__(
            normalize=normalize,
            name=name,
            n_jobs=n_jobs,
            none_handling=none_handling,
            fill_value=fill_value,
        )


    def _transform_single(self, value: RDKitMol) -> Optional[npt.NDArray["cp_data.MoleculeDatapoint"]]:
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
        smiles = Chem.MolToSmiles(value) # TODO this should not be needed in case chemprop includes a class which uses RDKit
        vec = np.array(
            [cp_data.MoleculeDatapoint(smiles, None)]
        )
        if np.any(np.isnan(vec)):
            return None
        return vec
