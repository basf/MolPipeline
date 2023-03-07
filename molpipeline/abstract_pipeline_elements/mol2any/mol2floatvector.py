"""Abstract classes for transforming rdkit molecules to float vectors."""
from __future__ import annotations

import abc
from typing import Iterable

import numpy as np
import numpy.typing as npt
from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement
from molpipeline.utils.multi_proc import wrap_parallelizable_task


class MolToDescriptorPipelineElement(MolToAnyPipelineElement):
    """PipelineElement which generates a matrix from descriptor-vectors of each molecule."""

    _normalize: bool
    _mean: npt.NDArray[np.float_]
    _std: npt.NDArray[np.float_]

    def __init__(
        self,
        normalize: bool = True,
        name: str = "MolToDescriptorPipelineElement",
        n_jobs: int = 1,
    ) -> None:
        """Initialize MolToDescriptorPipelineElement.

        Parameters
        ----------
        normalize: bool
        name: str
        n_jobs: int
        """
        super().__init__(name=name, n_jobs=n_jobs)
        self._normalize = normalize

    @property
    @abc.abstractmethod
    def n_features(self) -> int:
        """Return the number of features."""

    @staticmethod
    def assemble_output(
        value_list: Iterable[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """Transform output of all transform_single operations to matrix."""
        return np.vstack(list(value_list))

    def fit(self, value_list: list[Chem.Mol]) -> None:
        """Fit object to data."""
        self.fit_transform(value_list)

    def fit_transform(self, value_list: list[Chem.Mol]) -> npt.NDArray[np.float_]:
        """Fit object to data and return the accordingly transformed data."""
        array_list = wrap_parallelizable_task(
            self._transform_single, value_list, self.n_jobs
        )
        value_matrix = self.assemble_output(array_list)
        self._mean = np.nanmean(value_matrix, axis=0)
        self._std = np.nanstd(value_matrix, axis=0)
        self._std[np.where(self._std == 0)] = 1
        return self._normalize_matrix(value_matrix)

    def _normalize_matrix(
        self, value_matrix: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        if self._normalize:
            scaled_matrix = (value_matrix - self._mean) / self._std
            return scaled_matrix
        return value_matrix

    def transform(self, value_list: list[Chem.Mol]) -> npt.NDArray[np.float_]:
        """Transform the list of molecules to sparse matrix."""
        return self.assemble_output(super().transform(value_list))

    def transform_single(self, value: Chem.Mol) -> npt.NDArray[np.float_]:
        """Normalize _transform_single if required."""
        if self._normalize:
            return self._normalize_matrix(self._transform_single(value))
        return self._transform_single(value)

    @abc.abstractmethod
    def _transform_single(self, value: Chem.Mol) -> npt.NDArray[np.float_]:
        """Transform mol to dict, where items encode columns indices and values, respectively.

        Parameters
        ----------
        value: Chem.Mol

        Returns
        -------
        npt.NDArray[np.float_]
            Vector with descriptor values of molecule.
        """
