"""Module for custom filter functionality."""

from __future__ import annotations

from typing import Any, Callable, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
)
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol


class CustomFilter(_MolToMolPipelineElement):
    """Filters molecules based on a custom boolean function. Elements not passing the filter will be set to InvalidInstances."""

    def __init__(
        self,
        func: Callable[[RDKitMol], bool],
        name: str = "CustomFilter",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize CustomFilter.

        Parameters
        ----------
        func : Callable[[RDKitMol], bool]
            custom function to filter molecules
        name : str, optional
            name of the element, by default "CustomFilter"
        n_jobs : int, optional
            number of jobs to use, by default 1
        uuid : str, optional
            uuid of the element, by default None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.func = func

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Pretransform single value.

        Applies the custom boolean function to the molecule.

        Parameters
        ----------
        value : RDKitMol
            input value

        Returns
        -------
        OptionalMol
            output value
        """
        if self.func(value):
            return value
        return InvalidInstance(
            self.uuid,
            f"Molecule does not match filter from {self.name}",
            self.name,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters of CustomFilter.

        Parameters
        ----------
        deep: bool, optional (default: True)
            If True, return the parameters of all subobjects that are PipelineElements.

        Returns
        -------
        dict[str, Any]
            Parameters of CustomFilter.
        """
        params = super().get_params(deep=deep)
        if deep:
            params["func"] = self.func
        else:
            params["func"] = self.func
        return params

    def set_params(self, **parameters: dict[str, Any]) -> Self:
        """Set parameters of CustomFilter.

        Parameters
        ----------
        parameters: dict[str, Any]
            Parameters to set.

        Returns
        -------
        Self
            Self.
        """
        parameter_copy = dict(parameters)
        if "func" in parameter_copy:
            self.func = parameter_copy.pop("func")  # type: ignore
        super().set_params(**parameter_copy)
        return self
