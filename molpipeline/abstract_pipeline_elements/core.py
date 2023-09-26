"""All abstract classes later pipeline elements inherit from."""
from __future__ import annotations  # for all the python 3.8 users out there.

import abc
import copy
from typing import Any, Iterable, NamedTuple, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from uuid import uuid4
import numpy as np
from rdkit.Chem import Mol as RDKitMol  # pylint: disable=no-name-in-module

from molpipeline.utils.multi_proc import check_available_cores, wrap_parallelizable_task


class InvalidInstance(NamedTuple):
    """Object which is returned when an instance cannot be processed."""

    element_id: str
    message: str
    element_name: Optional[str] = None

    def __repr__(self) -> str:
        """Return string representation of InvalidInstance."""
        return (
            f"InvalidInstance({self.element_name or self.element_id}, {self.message})"
        )


OptionalMol = Union[RDKitMol, InvalidInstance]


class RemovedInstance:  # pylint: disable=too-few-public-methods
    """Object which is returned by a NoneFilter if an Invalid instance was removed."""

    def __init__(self, filter_element_id: str, message: Optional[str] = None) -> None:
        """Initialize RemovedInstance.

        Parameters
        ----------
        filter_element: str
            FilterElement which removed the molecule.
        message: Optional[str]
            Optional message why the molecule was removed.

        Returns
        -------
        None
        """
        self.filter_element_id = filter_element_id
        self.message = message

    def __repr__(self) -> str:
        """Return string representation of RemovedInstance."""
        return f"RemovedInstance({self.filter_element_id}, {self.message})"


class ABCPipelineElement(abc.ABC):
    """Ancestor of all PipelineElements."""

    name: str
    _requires_fitting: bool = False
    uuid: str

    def __init__(
        self,
        name: str = "ABCPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
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
        if uuid is None:
            self.uuid = str(uuid4())
        else:
            self.uuid = uuid

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
                "uuid": copy.copy(self.uuid),
            }

        return {
            "name": self.name,
            "n_jobs": self.n_jobs,
            "uuid": self.uuid,
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
                raise ValueError(
                    f"Cannot set attribute {att_name} on {self.__class__.__name__}"
                )
            setattr(self, att_name, att_value)
        return self

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

    @property
    def requires_fitting(self) -> bool:
        """Return whether the object requires fitting or not."""
        return self._requires_fitting

    def finish(self) -> None:
        """Inform object that iteration has been finished. Does in most cases nothing.

        Called after all transform singles have been processed. From MolPipeline
        """

    def fit_to_result(self, values: Any) -> Self:  # pylint: disable=unused-argument
        """Fit object to result of transformed values.

        Fit object to the result of the transform function. This is useful catching nones and removed molecules.

        Parameters
        ----------
        values: Any
            List of molecule representations.

        Returns
        -------
        Self
            Fitted object.
        """
        return self

    @abc.abstractmethod
    def fit_transform(self, values: Any) -> Any:
        """Apply fit function and subsequently transform the input.

        Parameters
        ----------
        values: Any
            Apply transformation specified in transform_single to all molecules in the value_list.

        Returns
        -------
        Any
            List of instances in new representation.
        """

    @abc.abstractmethod
    def transform(self, values: Any) -> Any:
        """Transform input_values according to object rules.

        Parameters
        ----------
        values: Any
            Iterable of molecule representations (SMILES, MolBlocks RDKit Molecules, PhysChem vectors etc.).
            Input depends on the concrete PipelineElement.

        Returns
        -------
        Any
            Transformed input_values.
        """

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


class TransformingPipelineElement(ABCPipelineElement):
    """Ancestor of all PipelineElements."""

    _input_type: type
    _output_type: type
    name: str

    def __init__(
        self,
        name: str = "ABCPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize ABCPipelineElement.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of cores used for processing.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
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
        _ = self.fit_transform(value_list)
        return self

    def fit_to_result(self, values: Any) -> Self:
        """Fit object to result of transformed values.

        Fit object to the result of the transform function. This is useful catching nones and removed molecules.

        Parameters
        ----------
        values: Any
            List of molecule representations.

        Returns
        -------
        Self
            Fitted object.
        """
        self._is_fitted = True
        return super().fit_to_result(values)

    def fit_transform(self, values: Any) -> Any:
        """Apply fit function and subsequently transform the input.

        Parameters
        ----------
        values: Any
            Apply transformation specified in transform_single to all molecules in the value_list.

        Returns
        -------
        Any
            List of molecules in new representation.
        """
        self._is_fitted = True
        if self.requires_fitting:
            pre_value_list = self.pretransform(values)
            self.fit_to_result(pre_value_list)
            output_list = self.finalize_list(pre_value_list)
            if hasattr(self, "assemble_output"):
                return self.assemble_output(output_list)
        return self.transform(values)

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
        pre_value = self.pretransform_single(value)
        if isinstance(pre_value, InvalidInstance):
            return pre_value
        return self.finalize_single(pre_value)

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

    @abc.abstractmethod
    def pretransform_single(self, value: Any) -> Any:
        """Transform the instance, but skips parameters learned during fitting.

        This is the first step for the full transformation.
        It is followed by the finalize_single method and assemble output which collects all single transformations.
        These functions are split as they need to be accessed separately from outside the object.

        Parameters
        ----------
        value: Any
            Value to be pretransformed.

        Returns
        -------
        Any
            Pretransformed value. (Skips applying parameters learned during fitting)
        """

    def finalize_single(self, value: Any) -> Any:
        """Apply parameters learned during fitting to a single instance.

        Parameters
        ----------
        value: Any
            Value obtained from pretransform_single.

        Returns
        -------
        Any
            Finalized value.
        """
        # Final cleanup of the molecule
        if isinstance(value, RDKitMol):
            if value.GetNumAtoms() == 0:
                return InvalidInstance(self.uuid, "Empty molecule", self.name)
        return value

    def pretransform(self, value_list: Iterable[Any]) -> list[Any]:
        """Transform input_values according to object rules without fitting specifics.

        Parameters
        ----------
        value_list: Iterable[Any]
            Iterable of instances to be pretransformed.

        Returns
        -------
        list[Any]
            Transformed input_values.
        """
        output_values = wrap_parallelizable_task(
            self.pretransform_single, value_list, self.n_jobs
        )
        return output_values

    def finalize_list(self, value_list: Iterable[Any]) -> list[Any]:
        """Transform list of values according to parameters learned during fitting.

        Parameters
        ----------
        value_list:  Iterable[Any]
            List of values to be transformed.

        Returns
        -------
        list[Any]
            List of transformed values.
        """
        output_values = wrap_parallelizable_task(
            self.finalize_single, value_list, self.n_jobs
        )
        return output_values

    def transform(self, values: Any) -> Any:
        """Transform input_values according to object rules.

        Parameters
        ----------
        values: Any
            Iterable of molecule representations (SMILES, MolBlocks RDKit Molecules, PhysChem vectors etc.).
            Input depends on the concrete PipelineElement.

        Returns
        -------
        Any
            Transformed input_values.
        """
        output_rows = self.pretransform(values)
        output_rows = self.finalize_list(output_rows)
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


class MolToMolPipelineElement(TransformingPipelineElement, abc.ABC):
    """Abstract PipelineElement where input and outputs are molecules."""

    _input_type = RDKitMol
    _output_type = RDKitMol

    def __init__(
        self,
        name: str = "MolToMolPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToMolPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def transform(self, values: list[OptionalMol]) -> list[OptionalMol]:
        """Transform list of molecules to list of molecules.

        Parameters
        ----------
        values: list[OptionalMol]
            List of molecules to be transformed.

        Returns
        -------
        list[OptionalMol]
            List of molecules or InvalidInstances, if corresponding transformation was not successful.
        """
        mol_list: list[OptionalMol] = super().transform(values)  # Stupid mypy...
        return mol_list

    def transform_single(self, value: OptionalMol) -> OptionalMol:
        """Apply pretransform_single and finalize_single in one step.

        Parameters
        ----------
        value: OptionalMol
            Molecule to be transformed.

        Returns
        -------
        OptionalMol
            Transformed molecule if transformation was successful, else InvalidInstance.
        """
        return super().transform_single(value)

    @abc.abstractmethod
    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Transform the molecule to another molecule object.

        Do not apply parameters learned during fitting.

        Parameters
        ----------
        value: RDKitMol
            Molecule to be transformed.

        Returns
        -------
        OptionalMol
            Transformed molecule if transformation was successful, else InvalidInstance.
        """


class AnyToMolPipelineElement(TransformingPipelineElement, abc.ABC):
    """Abstract PipelineElement which creates molecules from different inputs."""

    _output_type = RDKitMol

    def __init__(
        self,
        name: str = "AnyToMolPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize AnyToMolPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def transform(self, values: Any) -> list[OptionalMol]:
        """Transform list of instances to list of molecules.

        Parameters
        ----------
        values: Any
            Instances to be transformed to a list of molecules.

        Returns
        -------
        list[OptionalMol]
            List of molecules or InvalidInstances, if corresponding representation was invalid.
        """
        mol_list: list[OptionalMol] = super().transform(values)  # Stupid mypy...
        return mol_list

    @abc.abstractmethod
    def pretransform_single(self, value: Any) -> OptionalMol:
        """Transform the instance to a molecule, but skip parameters learned during fitting.

        Parameters
        ----------
        value: Any
            Representation to be transformed to a molecule.

        Returns
        -------
        OptionalMol
            Obtained molecule if valid representation, else InvalidInstance.
        """


class MolToAnyPipelineElement(TransformingPipelineElement, abc.ABC):
    """Abstract PipelineElement which creates molecules from different inputs."""

    _input_type = RDKitMol

    def __init__(
        self,
        name: str = "MolToAnyPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToAnyPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    @abc.abstractmethod
    def pretransform_single(self, value: RDKitMol) -> Any:
        """Transform the molecule, but skip parameters learned during fitting.

        Parameters
        ----------
        value: RDKitMOl
            Molecule to be transformed.

        Returns
        -------
        Any
            Transformed molecule.
        """
