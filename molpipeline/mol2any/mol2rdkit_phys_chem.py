"""Classes for encoding molecules as phys-chem vector."""

# pylint: disable=too-many-arguments

from __future__ import annotations

from typing import Any, Callable, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy

import numpy as np
import numpy.typing as npt
from loguru import logger
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.abstract_pipeline_elements.mol2any.mol2floatvector import (
    MolToDescriptorPipelineElement,
)
from molpipeline.utils.molpipeline_types import AnyTransformer, RDKitMol

RDKIT_DESCRIPTOR_DICT: dict[str, Callable[[Chem.Mol], float]]
RDKIT_DESCRIPTOR_DICT = dict(Descriptors.descList)

# MolWt is removed as ExactMolWt is already included.
# Ipc is removed because it causes trouble with numpy.
DEFAULT_DESCRIPTORS = [
    name for name in RDKIT_DESCRIPTOR_DICT if name not in ["MolWt", "Ipc"]
]


class MolToRDKitPhysChem(MolToDescriptorPipelineElement):
    """PipelineElement for creating a Descriptor vector based on RDKit phys-chem properties."""

    _descriptor_list: list[str]

    def __init__(
        self,
        descriptor_list: Optional[list[str]] = None,
        return_with_errors: bool = False,
        standardizer: Optional[AnyTransformer] = StandardScaler(),
        log_exceptions: bool = True,
        name: str = "Mol2RDKitPhysChem",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToRDKitPhysChem.

        Parameters
        ----------
        descriptor_list: Optional[list[str]], optional (default=None)
            List of descriptor names to calculate. If None, DEFAULT_DESCRIPTORS are used.
        return_with_errors: bool, optional (default = False)
            If False, the pipeline element will return an InvalidInstance if any error occurs during calculations.
            If True, the pipeline element will return a vector with NaN values for failed descriptor calculations.
        standardizer: Optional[AnyTransformer], optional (default=StandardScaler())
            Standardizer to use.
        log_exceptions: bool, optional (default=True)
            If True, traceback of exceptions occurring during descriptor calculation are logged.
        name: str, optional (default="Mol2RDKitPhysChem")
            Name of the PipelineElement.
        n_jobs: int, optional (default=1)
            Number of jobs to use for parallelization.
        uuid: Optional[str], optional (default=None)
            UUID of the PipelineElement. If None, a new UUID is generated.
        """
        self.descriptor_list = descriptor_list  # type: ignore
        self._feature_names = self._descriptor_list
        self._return_with_errors = return_with_errors
        self._log_exceptions = log_exceptions
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
        """Return a copy of the descriptor list. Alias of `feature_names`."""
        return self._descriptor_list[:]

    @descriptor_list.setter
    def descriptor_list(self, descriptor_list: list[str] | None) -> None:
        """Set the descriptor list.

        Parameters
        ----------
        descriptor_list: list[str] | None
            List of descriptor names to calculate. If None, DEFAULT_DESCRIPTORS are used.

        Raises
        ------
        ValueError
            If an unknown descriptor name is used.
        """
        if descriptor_list is None or descriptor_list is DEFAULT_DESCRIPTORS:
            # if None or DEFAULT_DESCRIPTORS are used, set the default descriptors
            self._descriptor_list = DEFAULT_DESCRIPTORS
        else:
            # check all user defined descriptors are valid
            for descriptor_name in descriptor_list:
                if descriptor_name not in RDKIT_DESCRIPTOR_DICT:
                    raise ValueError(
                        f"Unknown descriptor function with name: {descriptor_name}"
                    )
            self._descriptor_list = descriptor_list

    def pretransform_single(
        self, value: RDKitMol
    ) -> Union[npt.NDArray[np.float64], InvalidInstance]:
        """Transform a single molecule to a descriptor vector.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule to transform.

        Returns
        -------
        Optional[npt.NDArray[np.float64]]
            Descriptor vector for given molecule. None if calculation failed.
        """
        vec = np.full((len(self._descriptor_list),), np.nan)
        log_block = rdBase.BlockLogs()  # pylint: disable=unused-variable
        for i, name in enumerate(self._descriptor_list):
            descriptor_func = RDKIT_DESCRIPTOR_DICT[name]
            try:
                vec[i] = descriptor_func(value)
            except Exception:  # pylint: disable=broad-except
                if self._log_exceptions:
                    logger.exception(f"Failed calculating descriptor: {name}")
        del log_block
        if not self._return_with_errors and np.any(np.isnan(vec)):
            return InvalidInstance(self.uuid, "NaN in descriptor vector", self.name)
        return vec

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the parameters of the pipeline element.

        Parameters
        ----------
        deep: bool
            If true create a deep copy of the parameters

        Returns
        -------
        dict[str, Any]
            Parameter of the pipeline element.
        """
        parent_dict = dict(super().get_params(deep=deep))
        if deep:
            parent_dict["descriptor_list"] = copy.deepcopy(self._descriptor_list)
            parent_dict["return_with_errors"] = copy.deepcopy(self._return_with_errors)
            parent_dict["log_exceptions"] = copy.deepcopy(self._log_exceptions)
        else:
            parent_dict["descriptor_list"] = self._descriptor_list
            parent_dict["return_with_errors"] = self._return_with_errors
            parent_dict["log_exceptions"] = self._log_exceptions
        return parent_dict

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: Any
            Parameters to set

        Returns
        -------
        Self
            Self
        """
        parameters_shallow_copy = dict(parameters)
        params_list = ["descriptor_list", "return_with_errors", "log_exceptions"]
        for param_name in params_list:
            if param_name in parameters:
                setattr(self, f"_{param_name}", parameters[param_name])
                parameters_shallow_copy.pop(param_name)
        super().set_params(**parameters_shallow_copy)
        return self
