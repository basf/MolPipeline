"""Pipeline element to calculate Mordred descriptors."""

import copy
import warnings
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt

try:
    from mordred import Calculator, descriptors
    from mordred._base.util import (
        is_missing,  # noqa: PLC2701  T
    )

    MORDRED_DESCRIPTOR_DICT = {
        str(desc): desc for desc in Calculator(descriptors, ignore_3D=True).descriptors
    }

except ImportError:
    warnings.warn(
        "mordredcommunity not installed. MolToMordred will not work.",
        ImportWarning,
        stacklevel=2,
    )
    MORDRED_DESCRIPTOR_DICT = {}

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.abstract_pipeline_elements.mol2any import (
    MolToDescriptorPipelineElement,
)
from molpipeline.utils.molpipeline_types import AnyTransformer, RDKitMol

MORDRED_DESCRIPTORS = list(MORDRED_DESCRIPTOR_DICT.keys())

# Exclude descriptors which are usually nan for >95% of molecules.
EXCLUDE_DESCRIPTORS = {
    "MAXssBe",
    "MAXsLi",
    "MAXssssBe",
    "MINssssGe",
    "MINsssGeH",
    "MINssGeH2",
    "MINsAsH2",
    "MINsssdAs",
    "MINsssAs",
    "MINssAsH",
    "MINsGeH3",
    "MINsssssAs",
    "MINssPbH2",
    "MINsssPbH",
    "MAXsGeH3",
    "MAXssGeH2",
    "MINssBe",
    "MINsLi",
    "MAXssssPb",
    "MAXsssPbH",
    "MAXsssdAs",
    "MAXsssAs",
    "MAXsSnH3",
    "MAXsssssAs",
    "MAXsssSnH",
    "MAXssSnH2",
    "MINsSnH3",
    "MINssssPb",
    "MAXssPbH2",
    "MAXssAsH",
    "MINsPbH3",
    "MINssssSn",
    "MINsssSnH",
    "MINssSnH2",
    "MAXsPbH3",
    "MAXssssSn",
    "MINssssBe",
    "MAXsssGeH",
    "MAXsAsH2",
    "MAXssssGe",
    "MAXddssSe",
    "MINddssSe",
    "MAXsSeH",
    "MINsSeH",
    "MINsSiH3",
    "MAXsSiH3",
    "MINssSiH2",
    "MAXssSiH2",
    "MINsNH3",
    "MINdssSe",
    "MAXsNH3",
    "MAXdssSe",
    "MINsPH2",
    "MAXsPH2",
    "MINssBH",
    "MAXssBH",
    "MINssPH",
    "MAXssPH",
    "MAXdSe",
    "MINdSe",
    "MAXssNH2",
    "MINssNH2",
    "MAXsssSiH",
    "MINsssSiH",
    "MINsssssP",
    "MAXsssssP",
    "MAXsssNH",
    "MINsssNH",
    "MAXaaSe",
    "MINaaSe",
    "MAXssSe",
    "MINssSe",
    "MAXssssB",
    "MINssssB",
    "MAXsssP",
    "MINsssP",
    "MAXddC",
    "MINddC",
    "MINsssB",
    "MAXsssB",
    "MINsSH",
    "MAXsSH",
    "MAXssssSi",
    "MINssssSi",
    "MAXssssN",
    "MINssssN",
    "MAXdssS",
    "MINdssS",
    "MINsI",
    "MAXsI",
    "MAXtCH",
    "MINtCH",
    "MAXdNH",
    "MINdNH",
    "MAXdsssP",
    "MINdsssP",
    "MAXdCH2",
    "MINdCH2",
    "MDEN-11",
    "MINdS",
    "MAXdS",
}

DEFAULT_DESCRIPTORS = [
    desc for desc in MORDRED_DESCRIPTORS if desc not in EXCLUDE_DESCRIPTORS
]


class MolToMordred(MolToDescriptorPipelineElement):
    """Pipeline element to calculate Mordred descriptors."""

    _descriptor_list: list[str]

    def __init__(
        self,
        descriptor_list: list[str] | None = None,
        return_with_errors: bool = False,
        standardizer: Literal["default"] | AnyTransformer | None = "default",
        return_dtype: type = np.float64,
        name: str = "MolToMordred",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize the MolToMordred pipeline element.

        Parameters
        ----------
        descriptor_list: list[str] | None
            List of descriptor names to calculate. If None, DEFAULT_DESCRIPTORS are
            used.
        return_with_errors: bool
            If True, return descriptor vectors even if some descriptors failed to
            calculate. Failed descriptors will be set to NaN.
        standardizer: AnyTransformer | None
            Standardizer to apply to the descriptor vectors. If None, no standardization
            is applied. If "default", a StandardScaler is used.
        return_dtype: type
            Data type of the returned descriptor vectors. Default is np.float64.
        name: str
            Name of the pipeline element.
        n_jobs: int
            Number of jobs to use for descriptor calculation.
        uuid: str | None
            UUID of the pipeline element.

        Raises
        ------
        ValueError
            If return_dtype is not np.float64 or np.float32.

        """
        self.descriptor_list = descriptor_list  # type: ignore
        self._feature_names = self._descriptor_list
        self._return_with_errors = return_with_errors
        if return_dtype not in {np.float64, np.float32}:
            raise ValueError("return_dtype must be np.float64 or np.float32")
        self._return_dtype = return_dtype
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
            List of descriptor names to calculate. If None, DEFAULT_DESCRIPTORS are
            used.

        Raises
        ------
        ValueError
            If an unknown descriptor name is used.
        ValueError
            If an empty descriptor_list is used.

        """
        if descriptor_list is None or descriptor_list is DEFAULT_DESCRIPTORS:
            # if None or DEFAULT_DESCRIPTORS are used, set the default descriptors
            self._descriptor_list = DEFAULT_DESCRIPTORS
        elif len(descriptor_list) == 0:
            raise ValueError(
                "Empty descriptor_list is not allowed."
                " Use None for default descriptors.",
            )
        else:
            # check all user defined descriptors are valid
            for descriptor_name in descriptor_list:
                if descriptor_name not in MORDRED_DESCRIPTORS:
                    raise ValueError(
                        f"Unknown descriptor function with name: {descriptor_name}",
                    )
            self._descriptor_list = descriptor_list
        # set calculator with the new descriptor list
        self._calc = Calculator(
            [MORDRED_DESCRIPTOR_DICT[d] for d in self._descriptor_list],
            ignore_3D=False,
        )

    def pretransform_single(
        self,
        value: RDKitMol,
    ) -> npt.NDArray[np.float64] | InvalidInstance:
        """Transform a single molecule to a descriptor vector.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule to transform.

        Returns
        -------
        npt.NDArray[np.float64] | InvalidInstance
            Descriptor vector for given molecule.
            Failure is indicated by an InvalidInstance.

        """
        # TODO set _Name to "" attribute in mol? Otherwise mordred will do an expensive
        #  MolToSmiles(RemoveHs(…)) internally just to add some name.
        # TODO having a Calculator for each worker might lead to RAM issues when n_jobs
        #  is large. But with the sklearn logic there is no way around that, I think.
        result = self._calc(value)
        vec = np.array(
            [np.nan if is_missing(v) else v for v in result],
            dtype=self._return_dtype,
        )
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
            parent_dict["return_dtype"] = self._return_dtype
        else:
            parent_dict["descriptor_list"] = self._descriptor_list
            parent_dict["return_with_errors"] = self._return_with_errors
            parent_dict["return_dtype"] = self._return_dtype
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
        if "descriptor_list" in parameters:
            # use the setter to validate the descriptor list and update the calculator
            self.descriptor_list = parameters["descriptor_list"]
            parameters_shallow_copy.pop("descriptor_list")
        params_list = ["return_with_errors", "return_dtype"]
        for param_name in params_list:
            if param_name in parameters:
                setattr(self, f"_{param_name}", parameters[param_name])
                parameters_shallow_copy.pop(param_name)
        super().set_params(**parameters_shallow_copy)
        return self
