"""Mol to chemical element count vector."""

import copy
from collections import Counter
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from rdkit.Chem import GetPeriodicTable

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.abstract_pipeline_elements.mol2any.mol2floatvector import (
    MolToDescriptorPipelineElement,
)
from molpipeline.utils.molpipeline_types import AnyTransformer, RDKitMol

# Element numbers to count. Per default, all elements in the periodic table up to
# oganesson (118) are counted.
_MAX_CHEMICAL_ELEMENT_NUMBER = 118
_DEFAULT_ELEMENTS_TO_COUNT: list[int] = list(range(1, _MAX_CHEMICAL_ELEMENT_NUMBER + 1))


class MolToElementCount(MolToDescriptorPipelineElement):
    """PipelineElement for counting chemical elements in molecules."""

    def __init__(
        self,
        element_list: list[int] | None = None,
        standardizer: Literal["default"] | AnyTransformer | None = "default",
        name: str = "MolToElementCount",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize MolToElementCount.

        Parameters
        ----------
        element_list: list[int] | None, default=None
            List of atomic numbers to count. If None, all elements from
            hydrogen (1) to oganesson (118) are counted.
        standardizer: AnyTransformer | None, default=StandardScaler()
            Used for post-processing the output, if not None.
        name: str, default="MolToElementCount"
            Name of the PipelineElement.
        n_jobs: int, default=1
            Number of jobs to use for parallelization.
        uuid: str | None, optional
            UUID of the PipelineElement.

        """
        super().__init__(  # pylint: disable=duplicate-code
            standardizer=standardizer,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        self._feature_names: list[str] = []
        # Use the property setter for validation and feature name initialization.
        self.element_list = element_list

    @property
    def element_list(self) -> list[int]:
        """Get the element list."""
        return self._element_list[:]

    @element_list.setter
    def element_list(self, element_list: list[int] | None) -> None:
        """Set the element list.

        Parameters
        ----------
        element_list: list[int] | None
            List of atomic numbers to count. If None, all elements from
            hydrogen (1) to oganesson (118) are counted.

        Raises
        ------
        ValueError
            If any element is not an integer between 1 and 118.

        """
        if element_list is None or element_list is _DEFAULT_ELEMENTS_TO_COUNT:
            element_list = _DEFAULT_ELEMENTS_TO_COUNT
        elif not all(
            isinstance(element, int) and 1 <= element <= _MAX_CHEMICAL_ELEMENT_NUMBER
            for element in element_list
        ):
            msg = "Element list must contain integers between 1 and 118."
            raise ValueError(msg)
        self._element_list = element_list
        self._feature_names = [
            f"Count_{GetPeriodicTable().GetElementSymbol(atomic_num)}"
            for atomic_num in self._element_list
        ]
        self._count_hydrogen = 1 in self._element_list

    @property
    def n_features(self) -> int:
        """Return the number of features."""
        return len(self._element_list)

    def pretransform_single(
        self,
        value: RDKitMol,
    ) -> npt.NDArray[np.float64] | InvalidInstance:
        """Transform a single molecule to an element count vector.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule to transform.

        Returns
        -------
        npt.NDArray[np.float64] | InvalidInstance
            Array with element counts for the given molecule.

        """
        elements_list = [atom.GetAtomicNum() for atom in value.GetAtoms()]
        elements_counter = Counter(elements_list)
        if self._count_hydrogen:
            # Hydrogens are special because they can be implicit in the molecule
            # datastructure. We need to add the implicit hydrogens to the count.
            elements_counter[1] += sum(a.GetTotalNumHs() for a in value.GetAtoms())
        return np.array(
            [elements_counter.get(x, 0) for x in self._element_list],
            dtype=np.float64,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the parameters of the pipeline element.

        Parameters
        ----------
        deep: bool, default=True
            If True, create a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Parameters of the pipeline element.

        """
        parent_dict = dict(super().get_params(deep=deep))
        if deep:
            parent_dict["element_list"] = copy.deepcopy(self._element_list)
        else:
            parent_dict["element_list"] = self._element_list
        return parent_dict

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters.

        Parameters
        ----------
        parameters: Any
            Parameters to set.

        Returns
        -------
        Self
            Object with updated parameters.

        """
        parameters_shallow_copy = dict(parameters)
        if "element_list" in parameters_shallow_copy:
            self.element_list = parameters_shallow_copy.pop("element_list")
        super().set_params(**parameters_shallow_copy)
        return self
