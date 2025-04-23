"""Abstract classes for transforming rdkit molecules to float vectors."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from molpipeline.abstract_pipeline_elements.core import (
    InvalidInstance,
    MolToAnyPipelineElement,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from molpipeline.utils.molpipeline_types import RDKitMol


class MolToDescriptorPipelineElement(MolToAnyPipelineElement):
    """PipelineElement for generating descriptor-vectors."""

    _output_type = "float"
    _feature_names: list[str]

    def __init__(
        self,
        name: str = "MolToDescriptorPipelineElement",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize MolToDescriptorPipelineElement.

        Parameters
        ----------
        name: str, default='MolToDescriptorPipelineElement'
            Name of the PipelineElement.
        n_jobs: int, default=1
            Number of jobs to use for parallelization.
        uuid: str | None, optional
            UUID of the PipelineElement.

        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self._mean = None
        self._std = None

    @property
    @abc.abstractmethod
    def n_features(self) -> int:
        """Return the number of features."""

    @property
    def feature_names(self) -> list[str]:
        """Return a copy of the feature names."""
        return self._feature_names[:]

    def transform(self, values: list[RDKitMol]) -> npt.NDArray[np.float64]:
        """Transform the list of molecules to sparse matrix.

        Parameters
        ----------
        values: list[RDKitMol]
            List of RDKit molecules for which the descriptor vectors are calculated.

        Returns
        -------
        npt.NDArray[np.float64]
            Matrix with descriptor values of molecules.

        """
        descriptor_matrix: npt.NDArray[np.float64] = super().transform(values)
        return descriptor_matrix

    @abc.abstractmethod
    def pretransform_single(
        self,
        value: RDKitMol,
    ) -> npt.NDArray[np.float64] | InvalidInstance:
        """Transform mol to dict.

        Items encode columns indices and values, respectively.

        Parameters
        ----------
        value: Chem.Mol
            Molecule to be transformed to descriptor vector.

        Returns
        -------
        npt.NDArray[np.float64] | InvalidInstance
            Vector with descriptor values of molecule.

        """
