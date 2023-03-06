"""All abstract classes later pipeline elements inherit from."""
from __future__ import annotations  # for all the python 3.8 users out there.

import abc
from typing import Any
from rdkit import Chem

from molpipeline.utils.molpipe_types import OptionalMol
from molpipeline.utils.multi_proc import check_available_cores, wrap_parallelizable_task


class ABCPipelineElement(abc.ABC):
    """Ancestor of all PipelineElements."""

    _input_type: type
    _output_type: type
    name: str

    def __init__(self, name: str = "ABCPipelineElement", n_jobs: int = 1) -> None:
        """Initialize ABCPipelineElement.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of cores used for processing.
        """
        self.name = name
        self.n_jobs = n_jobs

    @property
    def input_type(self) -> type:
        """Return the input type."""
        return self._input_type

    @property
    def n_jobs(self) -> int:
        """Get the number of cores."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int) -> None:
        """Set the number of cores."""
        self._n_jobs = check_available_cores(n_jobs)

    @property
    def output_type(self) -> type:
        """Return the output type."""
        return self._output_type

    def fit(self, value_list: Any) -> None:
        """Fit object to input_values. Does often nothing."""

    def fit_transform(self, value_list: Any) -> Any:
        """Apply fit function and subsequently transform the input."""
        self.fit(value_list)
        return self.transform(value_list)

    def transform_single(self, value: Any) -> Any:
        """Transform the input to the new Output."""
        return self._transform_single(value)

    @abc.abstractmethod
    def transform(self, value_list: Any) -> Any:
        """Transform input_values according to object rules."""
        output_values = wrap_parallelizable_task(
            self.transform_single, value_list, self.n_jobs
        )
        self.finish()
        return output_values

    def finish(self) -> None:
        """Inform object that iteration has been finished. Does in most cases nothing.

        Called after all transform singles have been processed. From MolPipeline
        """

    @abc.abstractmethod
    def _transform_single(self, value: Any) -> Any:
        """Transform the molecule according to child dependent rules."""


class MolToMolPipelineElement(ABCPipelineElement, abc.ABC):
    """Abstract PipelineElement where input and outputs are molecules."""

    _input_type = Chem.Mol
    _output_type = Chem.Mol

    def __init__(self, name: str = "MolToMolPipelineElement", n_jobs: int = 1) -> None:
        """Initialize MolToMolPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs)

    def transform(self, value_list: list[OptionalMol]) -> list[OptionalMol]:
        """Transform list of molecules to list of molecules."""
        mol_list: list[OptionalMol] = super().transform(value_list)  # Stupid mypy...
        return mol_list

    def transform_single(self, value: OptionalMol) -> OptionalMol:
        """Wrap the transform_single method to handle Nones."""
        if not value:
            return None
        return self._transform_single(value)

    @abc.abstractmethod
    def _transform_single(self, value: Chem.Mol) -> OptionalMol:
        """Transform the molecule according to child dependent rules."""


class AnyToMolPipelineElement(ABCPipelineElement, abc.ABC):
    """Abstract PipelineElement which creates molecules from different inputs."""

    _output_type = Chem.Mol

    def __init__(self, name: str = "AnyToMolPipelineElement", n_jobs: int = 1) -> None:
        """Initialize AnyToMolPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs)

    def transform(self, value_list: Any) -> list[OptionalMol]:
        """Transform list of molecules to list of molecules."""
        mol_list: list[OptionalMol] = super().transform(value_list)  # Stupid mypy...
        return mol_list

    @abc.abstractmethod
    def _transform_single(self, value: Any) -> OptionalMol:
        """Transform the input specified in each child to molecules."""


class MolToAnyPipelineElement(ABCPipelineElement, abc.ABC):
    """Abstract PipelineElement which creates molecules from different inputs."""

    _input_type = Chem.Mol

    def __init__(self, name: str = "MolToAnyPipelineElement", n_jobs: int = 1) -> None:
        """Initialize MolToAnyPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs)

    @abc.abstractmethod
    def _transform_single(self, value: Chem.Mol) -> Any:
        """Transform the molecules to the input specified in each child."""
