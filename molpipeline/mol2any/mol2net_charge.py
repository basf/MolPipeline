"""MolToNetCharge pipeline element."""

from __future__ import annotations

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
        standardizer: AnyTransformer, optional
            Standardizer to use, by default StandardScaler()
        name: str, optional
            Name of the pipeline element, by default "MolToNetCharge"
        n_jobs: int, optional
            Number of jobs to run in parallel, by default 1
        uuid: str, optional
            UUID of the pipeline element, by default None
        """
        self._descriptor_list = ["NetCharge"]
        # pylint: disable=R0801
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
        net_charge = np.round(np.sum(atoms_contributions, keepdims=True))
        return net_charge
