"""All abstract classes later pipeline elements inherit from."""
from __future__ import annotations  # for all the python 3.8 users out there.

import abc
from typing import Any, Iterable, Literal

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from rdkit import Chem

from molpipeline.utils.molpipe_types import OptionalMol
from molpipeline.utils.multi_proc import check_available_cores, wrap_parallelizable_task
from molpipeline.utils.none_handling import NoneCollector

NoneHandlingOptions = Literal["raise", "record_remove", "fill_dummy"]


class ABCPipelineElement(abc.ABC):
    """Ancestor of all PipelineElements."""

    _input_type: type
    _output_type: type
    name: str

    def __init__(
        self,
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = None,
        name: str = "ABCPipelineElement",
        n_jobs: int = 1,
    ) -> None:
        """Initialize ABCPipelineElement.

        Parameters
        ----------
        none_handling: Literal["raise", "record_remove"]
            Behaviour when encountering None values, aka. unprocessable molecules.
        fill_value: Any
            value used for the NoneHandler.
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of cores used for processing.
        """
        self.name = name
        self.none_handling = none_handling
        self.n_jobs = n_jobs
        self.none_collector = NoneCollector(fill_value)

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> Self:
        """Create object from json dict."""
        specified_class = json_dict.pop("type")
        specified_module = json_dict.pop("module")
        if specified_module != cls.__module__:
            raise ValueError(f"Cannot create {cls.__name__} from {specified_module}")
        if specified_class != cls.__name__:
            raise ValueError(f"Cannot create {cls.__name__} from {specified_class}")
        return cls(**json_dict)

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
    def none_handling(self) -> NoneHandlingOptions:
        """Get string which determines the handling of nones."""
        return self._none_handling

    @none_handling.setter
    def none_handling(self, none_handling: NoneHandlingOptions) -> None:
        """Set string which determines the handling of nones."""
        valid_options = ["raise", "record_remove", "fill_dummy"]
        if none_handling not in valid_options:
            raise ValueError(
                f"{none_handling} is not a valid option. Please choose from f{valid_options}"
            )
        self._none_handling = none_handling

    @property
    def output_type(self) -> type:
        """Return the output type."""
        return self._output_type

    @property
    def params(self) -> dict[str, Any]:
        """Any parameter relevant for creating and exact copy."""
        return {
            "name": self.name,
            "none_handling": self.none_handling,
            "n_jobs": self.n_jobs,
            "fill_value": self.none_collector.fill_value,
        }

    @abc.abstractmethod
    def copy(self) -> ABCPipelineElement:
        """Copy the object."""

    def fit(self, value_list: Any) -> None:
        """Fit object to input_values. Does often nothing."""

    def fit_transform(self, value_list: Any) -> Any:
        """Apply fit function and subsequently transform the input."""
        self.fit(value_list)
        return self.transform(value_list)

    def transform_single(self, value: Any) -> Any:
        """Transform the input to the new Output."""
        return self._transform_single(value)

    def _apply_to_all(self, value_list: Any) -> Any:
        """Transform input_values according to object rules."""
        output_values = wrap_parallelizable_task(
            self.transform_single, value_list, self.n_jobs
        )
        return output_values

    def _catch_nones(self, value_list: list[Any]) -> list[Any]:
        none_rows = [idx for idx, row in enumerate(value_list) if row is None]
        if len(none_rows) > 0 and self.none_handling == "raise":
            raise ValueError(f"Encountered None for the following indices: {none_rows}")

        self.none_collector.none_indices = none_rows
        output_rows = [row for row in value_list if row is not None]
        return output_rows

    def assemble_output(self, value_list: Iterable[Any]) -> Any:
        """Aggregate rows, which in most cases is just return the list."""
        return list(value_list)

    def transform(self, value_list: Any) -> Any:
        """Transform input_values according to object rules."""
        output_rows = self._apply_to_all(value_list)
        output_rows = self._catch_nones(output_rows)
        output = self.assemble_output(output_rows)
        self.finish()
        if self.none_handling == "fill_dummy":
            return self.none_collector.fill_with_dummy(output)
        if self.none_handling == "record_remove":
            return output
        return output

    def to_json(self) -> dict[str, Any]:
        """Return a all definig attributes of object as dict."""
        json_dict = {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
        }
        json_dict.update(self.params)
        return json_dict

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

    def __init__(
        self,
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = None,
        name: str = "MolToMolPipelineElement",
        n_jobs: int = 1,
    ) -> None:
        """Initialize MolToMolPipelineElement."""
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )

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

    @property
    def params(self) -> dict[str, Any]:
        """Get object parameters relevant for copying the class."""
        return super().params


class AnyToMolPipelineElement(ABCPipelineElement, abc.ABC):
    """Abstract PipelineElement which creates molecules from different inputs."""

    _output_type = Chem.Mol

    def __init__(
        self,
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = None,
        name: str = "AnyToMolPipelineElement",
        n_jobs: int = 1,
    ) -> None:
        """Initialize AnyToMolPipelineElement."""
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )

    def transform(self, value_list: Any) -> list[OptionalMol]:
        """Transform list of molecules to list of molecules."""
        mol_list: list[OptionalMol] = super().transform(value_list)  # Stupid mypy...
        return mol_list

    @abc.abstractmethod
    def _transform_single(self, value: Any) -> OptionalMol:
        """Transform the input specified in each child to molecules."""

    @property
    def params(self) -> dict[str, Any]:
        """Get object parameters relevant for copying the class."""
        return super().params


class MolToAnyPipelineElement(ABCPipelineElement, abc.ABC):
    """Abstract PipelineElement which creates molecules from different inputs."""

    _input_type = Chem.Mol

    def __init__(
        self,
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = None,
        name: str = "MolToAnyPipelineElement",
        n_jobs: int = 1,
    ) -> None:
        """Initialize MolToAnyPipelineElement."""
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )

    @abc.abstractmethod
    def _transform_single(self, value: Chem.Mol) -> Any:
        """Transform the molecules to the input specified in each child."""

    @property
    def params(self) -> dict[str, Any]:
        """Get object parameters relevant for copying the class."""
        return super().params
