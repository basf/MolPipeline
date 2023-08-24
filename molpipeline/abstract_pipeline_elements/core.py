"""All abstract classes later pipeline elements inherit from."""
from __future__ import annotations  # for all the python 3.8 users out there.

import abc
import copy
from typing import Any, Iterable, Union, TYPE_CHECKING

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np

from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]
from molpipeline.utils.multi_proc import check_available_cores, wrap_parallelizable_task

if TYPE_CHECKING:  # Avoid circular imports
    from molpipeline.utils.none_handling import NoneFilter


class InvalidInstance:
    def __init__(self, element: TransformingPipelineElement, message: str) -> None:
        self.element = element
        self.message = message


class RemovedInstance:
    def __init__(self, filter_element: NoneFilter) -> None:
        self.filter_element = filter_element


OptionalMol = Union[RDKitMol, InvalidInstance]


class ABCPipelineElement(abc.ABC):
    """Ancestor of all PipelineElements."""

    name: str
    _requires_fitting: bool = False

    def __init__(
        self,
        name: str = "ABCPipelineElement",
        n_jobs: int = 1,
    ) -> None:
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

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> Self:
        """Create object from json dict.

        Parameters
        ----------
        json_dict: dict[str, Any]
            Json with parameters to initialize the object.

        Returns
        -------
        Self
            Object specified by json_dict.
        """
        json_dict_copy = dict(json_dict)
        specified_class = json_dict_copy.pop("__name__")
        specified_module = json_dict_copy.pop("__module__")
        if specified_module != cls.__module__:
            raise ValueError(f"Cannot create {cls.__name__} from {specified_module}")
        if specified_class != cls.__name__:
            raise ValueError(f"Cannot create {cls.__name__} from {specified_class}")
        additional_attributes = json_dict_copy.pop("additional_attributes", {})
        loaded_pipeline_element = cls(**json_dict_copy)
        for key, value in additional_attributes.items():
            if not hasattr(loaded_pipeline_element, key):
                raise ValueError(
                    f"Cannot set attribute {key} on {cls.__name__} from {specified_class}"
                )
            setattr(loaded_pipeline_element, key, value)
        return loaded_pipeline_element

    @property
    def additional_attributes(self) -> dict[str, Any]:
        """Any attribute relevant for recreating and exact copy, which is not a parameter.

        Returns
        -------
        dict[str, Any]
        """
        return {}

    @property
    def n_jobs(self) -> int:
        """Get the number of cores."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int) -> None:
        """Set the number of cores.

        Parameters
        ----------
        n_jobs: int
            Number of cores used for processing.

        Returns
        -------
        None
        """
        self._n_jobs = check_available_cores(n_jobs)

    @abc.abstractmethod
    def transform_single(self, value: Any) -> Any:
        """Transform a single value.

        Parameters
        ----------
        value: Any
            Value to be transformed.

        Returns
        -------
        Any
            Transformed value.
        """

    @property
    def requires_fitting(self) -> bool:
        """Return whether the object requires fitting or not."""
        return self._requires_fitting

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the parameters of the object.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Parameters of the object.
        """
        if deep:
            return {
                "name": copy.copy(self.name),
                "n_jobs": copy.copy(self.n_jobs),
            }
        else:
            return {
                "name": self.name,
                "n_jobs": self.n_jobs,
            }

    def set_params(self, parameters: dict[str, Any]) -> Self:
        """As the setter function cannot be assessed with super(), this method is implemented for inheritance.

        Parameters
        ----------
        parameters: dict[str, Any]
            Parameters to be set.

        Returns
        -------
        Self
            Self with updated parameters.
        """
        for att_name, att_value in parameters.items():
            if not hasattr(self, att_name):
                ValueError(
                    f"Cannot set attribute {att_name} on {self.__class__.__name__}"
                )
            setattr(self, att_name, att_value)
        return self

    def finish(self) -> None:
        """Inform object that iteration has been finished. Does in most cases nothing.

        Called after all transform singles have been processed. From MolPipeline
        """

    @abc.abstractmethod
    def fit_transform(self, value_list: Any) -> Any:
        """Apply fit function and subsequently transform the input.

        Parameters
        ----------
        value_list: Any
            Apply transformation specified in transform_single to all molecules in the value_list.

        Returns
        -------
        Any
            List of instances in new representation.
        """

    @abc.abstractmethod
    def transform(self, value_list: Any) -> Any:
        """Transform input_values according to object rules.

        Parameters
        ----------
        value_list: Any
            Iterable of molecule representations (SMILES, MolBlocks RDKit Molecules, PhysChem vectors etc.).
            Input depends on the concrete PipelineElement.

        Returns
        -------
        Any
            Transformed input_values.
        """


class TransformingPipelineElement(ABCPipelineElement):
    """Ancestor of all PipelineElements."""

    _input_type: type
    _output_type: type
    name: str

    def __init__(
        self,
        name: str = "ABCPipelineElement",
        n_jobs: int = 1,
    ) -> None:
        """Initialize ABCPipelineElement.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of cores used for processing.
        """
        super().__init__(name=name, n_jobs=n_jobs)
        self._is_fitted = False

    @property
    def input_type(self) -> type:
        """Return the input type."""
        return self._input_type

    @property
    def is_fitted(self) -> bool:
        """Return whether the object is fitted or not."""
        return self._is_fitted

    @property
    def output_type(self) -> type:
        """Return the output type."""
        return self._output_type

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the parameters of the object."""
        return self.get_params()

    @parameters.setter
    def parameters(self, parameters: dict[str, Any]) -> None:
        """Set the parameters of the object.

        Parameters
        ----------
        parameters: dict[str, Any]
            Object parameters as a dictionary.

        Returns
        -------
        None
        """
        self.set_params(parameters)

    def copy(self) -> Self:
        """Copy the object.

        Returns
        -------
        Self
            Copy of the object.
        """
        recreated_object = self.__class__(**self.parameters)
        for key, value in self.additional_attributes.items():
            if not hasattr(recreated_object, key):
                raise AssertionError(
                    f"Cannot set attribute {key} on {self.__class__.__name__}. This should not happen!"
                )
            setattr(recreated_object, key, copy.copy(value))
        return recreated_object

    def fit(self, value_list: Any) -> Self:
        """Fit object to input_values.

        Most objects might not need fitting, but it is implemented for consitency for all PipelineElements.

        Parameters
        ----------
        value_list: Any
            List of molecule representations.

        Returns
        -------
        Self
            Fitted object.
        """
        self._is_fitted = True
        return self

    def fit_transform(self, value_list: Any) -> Any:
        """Apply fit function and subsequently transform the input.

        Parameters
        ----------
        value_list: Any
            Apply transformation specified in transform_single to all molecules in the value_list.

        Returns
        -------
        Any
            List of molecules in new representation.
        """
        self.fit(value_list)
        return self.transform(value_list)

    def transform_single(self, value: Any) -> Any:
        """Transform a single molecule to the new representation.

        RemovedMolecule objects are passed without change, as no transformations are applicable.

        Parameters
        ----------
        value: Any
            Current representation of the molecule. (Eg. SMILES, RDKit Mol, ...)

        Returns
        -------
        Any
            New representation of the molecule. (Eg. SMILES, RDKit Mol, Descriptor-Vector, ...)
        """
        if isinstance(value, InvalidInstance):
            return value
        if isinstance(value, RDKitMol):
            if value.GetNumAtoms() == 0:
                return InvalidInstance(
                    message="No atoms remaining after transformation.",
                    element=self,
                )
        return self._transform_single(value)

    def _apply_to_all(self, value_list: Any) -> Any:
        """Transform input_values according to object rules."""
        output_values = wrap_parallelizable_task(
            self.transform_single, value_list, self.n_jobs
        )
        return output_values

    def assemble_output(self, value_list: Iterable[Any]) -> Any:
        """Aggregate rows, which in most cases is just return the list.

        Some representations might be better representd as a single object. For example a list of vectors can
        be transformed to a matrix.

        Parameters
        ----------
        value_list: Iterable[Any]
            Iterable of transformed rows.

        Returns
        -------
        Any
            Aggregated output. This can also be the original input.
        """
        return list(value_list)

    def transform(self, value_list: Any) -> Any:
        """Transform input_values according to object rules.

        Parameters
        ----------
        value_list: Any
            Iterable of molecule representations (SMILES, MolBlocks RDKit Molecules, PhysChem vectors etc.).
            Input depends on the concrete PipelineElement.

        Returns
        -------
        Any
            Transformed input_values.
        """
        output_rows = self._apply_to_all(value_list)
        output = self.assemble_output(output_rows)
        self.finish()
        return output

    def to_json(self) -> dict[str, Any]:
        """Return all defining attributes of object as dict.

        Returns
        -------
        dict[str, Any]
            A dictionary with all attributes necessary to initialize a object with same parameters.
        """
        json_dict: dict[str, Any] = {
            "__name__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
        }
        json_dict.update(self.parameters)
        if self.additional_attributes:
            adittional_attributes = {}
            for key, value in self.additional_attributes.items():
                if isinstance(value, np.ndarray):
                    adittional_attributes[key] = value.tolist()
                else:
                    adittional_attributes[key] = value
            json_dict["additional_attributes"] = adittional_attributes
        return json_dict

    @abc.abstractmethod
    def _transform_single(self, value: Any) -> Any:
        """Transform the molecule according to child dependent rules."""


class MolToMolPipelineElement(TransformingPipelineElement, abc.ABC):
    """Abstract PipelineElement where input and outputs are molecules."""

    _input_type = RDKitMol
    _output_type = RDKitMol

    def __init__(
        self,
        name: str = "MolToMolPipelineElement",
        n_jobs: int = 1,
    ) -> None:
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
    def _transform_single(self, value: RDKitMol) -> OptionalMol:
        """Transform the molecule according to child dependent rules."""


class AnyToMolPipelineElement(TransformingPipelineElement, abc.ABC):
    """Abstract PipelineElement which creates molecules from different inputs."""

    _output_type = RDKitMol

    def __init__(
        self,
        name: str = "AnyToMolPipelineElement",
        n_jobs: int = 1,
    ) -> None:
        """Initialize AnyToMolPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs)

    def transform(self, value_list: Any) -> list[OptionalMol]:
        """Transform list of molecules to list of molecules."""
        mol_list: list[OptionalMol] = super().transform(value_list)  # Stupid mypy...
        return mol_list

    @abc.abstractmethod
    def _transform_single(self, value: Any) -> OptionalMol:
        """Transform the input specified in each child to molecules."""


class MolToAnyPipelineElement(TransformingPipelineElement, abc.ABC):
    """Abstract PipelineElement which creates molecules from different inputs."""

    _input_type = RDKitMol

    def __init__(
        self,
        name: str = "MolToAnyPipelineElement",
        n_jobs: int = 1,
    ) -> None:
        """Initialize MolToAnyPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs)

    @abc.abstractmethod
    def _transform_single(self, value: RDKitMol) -> Any:
        """Transform the molecules to the input specified in each child."""
