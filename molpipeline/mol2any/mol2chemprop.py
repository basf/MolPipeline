"""Classes for encoding molecules as phys-chem vector."""

# pylint: disable=too-many-arguments

from __future__ import annotations

import warnings
from typing import Iterable, Optional

try:
    from chemprop.data import MoleculeDatapoint, MoleculeDataset
    from chemprop.featurizers.molecule import MoleculeFeaturizer
except ImportError:
    warnings.warn(
        "chemprop not installed. MolToChemprop will not work.",
        ImportWarning,
    )

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement
from molpipeline.utils.molpipeline_types import RDKitMol


class MolToChemprop(MolToAnyPipelineElement):
    """PipelineElement for creating a graph representation based on chemprop molecule classes."""

    featurizer_list: list[MoleculeFeaturizer] | None

    def __init__(
        self,
        featurizer_list: list[MoleculeFeaturizer] | None = None,
        name: str = "Mol2Chemprop",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToChemprop.

        Parameters
        ----------
        name: str
            Name of the pipeline element. Defaults to "Mol2Chemprop".
        n_jobs: int
            Number of parallel jobs to use. Defaults to 1.
        """
        self.featurizer_list = featurizer_list
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )

    def pretransform_single(self, value: RDKitMol) -> MoleculeDatapoint:
        """Transform a single molecule to a ChemProp MoleculeDataPoint.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule to transform.

        Returns
        -------
        MoleculeDatapoint
            Molecular representation used as input for ChemProp. None if transformation failed.
        """
        return MoleculeDatapoint(mol=value, mfs=self.featurizer_list)

    def assemble_output(
        self, value_list: Iterable[MoleculeDatapoint]
    ) -> MoleculeDataset:
        """Assemble the output from the parallelized pretransform_single.

        Parameters
        ----------
        value_list: Iterable
            List of transformed values.

        Returns
        -------
        Any
            Assembled output.
        """
        return MoleculeDataset(data=list(value_list))
