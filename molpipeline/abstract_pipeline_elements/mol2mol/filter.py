"""Abstract classes for filters."""

import abc
from typing import Any, Literal, Mapping, Optional, TypeAlias, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from molpipeline.abstract_pipeline_elements.core import (
    InvalidInstance,
    MolToMolPipelineElement,
    OptionalMol,
    RDKitMol,
)
from molpipeline.utils.value_conversions import (
    FloatCountRange,
    IntCountRange,
    IntOrIntCountRange,
    count_value_to_tuple,
)

# possible mode types for a KeepMatchesFilter:
# - "any" means one match is enough
# - "all" means all elements must be matched
FilterModeType: TypeAlias = Literal["any", "all"]


def _within_boundaries(
    lower_bound: Optional[float], upper_bound: Optional[float], value: float
) -> bool:
    """Check if a value is within the specified boundaries.

    Boundaries given as None are ignored.

    Parameters
    ----------
    lower_bound: Optional[float]
        Lower boundary.
    upper_bound: Optional[float]
        Upper boundary.
    value: float
        Value to check.

    Returns
    -------
    bool
        True if the value is within the boundaries, else False.
    """
    if lower_bound is not None and value < lower_bound:
        return False
    if upper_bound is not None and value > upper_bound:
        return False
    return True


class BaseKeepMatchesFilter(MolToMolPipelineElement, abc.ABC):
    """Filter to keep or remove molecules based on patterns.

    Notes
    -----
    There are four possible scenarios:
        - mode = "any" & keep_matches = True: Needs to match at least one filter element.
        - mode = "any" & keep_matches = False: Must not match any filter element.
        - mode = "all" & keep_matches = True: Needs to match all filter elements.
        - mode = "all" & keep_matches = False: Must not match all filter elements.
    """

    keep_matches: bool
    mode: FilterModeType

    def __init__(
        self,
        keep_matches: bool = True,
        mode: FilterModeType = "any",
        name: Optional[str] = None,
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize BasePatternsFilter.

        Parameters
        ----------
        keep_matches: bool, optional (default: True)
            If True, molecules containing the specified patterns are kept, else removed.
        mode: FilterModeType, optional (default: "any")
            If "any", at least one of the specified patterns must be present in the molecule.
            If "all", all of the specified patterns must be present in the molecule.
        name: Optional[str], optional (default: None)
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.keep_matches = keep_matches
        self.mode = mode

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters of BaseKeepMatchesFilter.

        Parameters
        ----------
        parameters: Any
            Parameters to set.

        Returns
        -------
        Self
            Self.
        """
        parameter_copy = dict(parameters)
        if "keep_matches" in parameter_copy:
            self.keep_matches = parameter_copy.pop("keep_matches")
        if "mode" in parameter_copy:
            self.mode = parameter_copy.pop("mode")
        super().set_params(**parameter_copy)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters of PatternFilter.

        Parameters
        ----------
        deep: bool, optional (default: True)
            If True, return the parameters of all subobjects that are PipelineElements.

        Returns
        -------
        dict[str, Any]
            Parameters of BaseKeepMatchesFilter.
        """
        params = super().get_params(deep=deep)
        params["keep_matches"] = self.keep_matches
        params["mode"] = self.mode
        return params

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Invalidate or validate molecule based on specified filter.

        There are four possible scenarios:
        - mode = "any" & keep_matches = True: Needs to match at least one filter element.
        - mode = "any" & keep_matches = False: Must not match any filter element.
        - mode = "all" & keep_matches = True: Needs to match all filter elements.
        - mode = "all" & keep_matches = False: Must not match all filter elements.

        Parameters
        ----------
        value: RDKitMol
            Molecule to check.

        Returns
        -------
        OptionalMol
            Molecule that matches defined filter elements, else InvalidInstance.
        """
        for filter_element, (min_count, max_count) in self.filter_elements.items():
            count = self._calculate_single_element_value(filter_element, value)
            if _within_boundaries(min_count, max_count, count):
                # For "any" mode we can return early if a match is found
                if self.mode == "any":
                    if not self.keep_matches:
                        value = InvalidInstance(
                            self.uuid,
                            f"Molecule contains forbidden filter element {filter_element}.",
                            self.name,
                        )
                    return value
            else:
                # For "all" mode we can return early if a match is not found
                if self.mode == "all":
                    if self.keep_matches:
                        value = InvalidInstance(
                            self.uuid,
                            f"Molecule does not contain required filter element {filter_element}.",
                            self.name,
                        )
                    return value

        # If this point is reached, no or all patterns were found
        # If mode is "any", finishing the loop means no match was found
        if self.mode == "any":
            if self.keep_matches:
                value = InvalidInstance(
                    self.uuid,
                    "Molecule does not match any of the required filter elements.",
                    self.name,
                )
            #  else: No match with forbidden filter elements was found, return original molecule
            return value

        if self.mode == "all":
            if not self.keep_matches:
                value = InvalidInstance(
                    self.uuid,
                    "Molecule matches all forbidden filter elements.",
                    self.name,
                )
            #  else: All required filter elements were found, return original molecule
            return value

        raise ValueError(f"Invalid mode: {self.mode}")

    @abc.abstractmethod
    def _calculate_single_element_value(
        self, filter_element: Any, value: RDKitMol
    ) -> float:
        """Calculate the value of a single match.

        Parameters
        ----------
        filter_element: Any
            Match case to calculate.
        value: RDKitMol
            Molecule to calculate the match for.

        Returns
        -------
        float
            Value of the match.
        """

    @property
    @abc.abstractmethod
    def filter_elements(
        self,
    ) -> Mapping[Any, FloatCountRange]:
        """Get filter elements as dict."""


class BasePatternsFilter(BaseKeepMatchesFilter, abc.ABC):
    """Filter to keep or remove molecules based on patterns.

    Notes
    -----
    There are four possible scenarios:
    - mode = "any" & keep_matches = True: Needs to match at least one filter element.
    - mode = "any" & keep_matches = False: Must not match any filter element.
    - mode = "all" & keep_matches = True: Needs to match all filter elements.
    - mode = "all" & keep_matches = False: Must not match all filter elements.
    """

    _patterns: dict[str, IntCountRange]

    def __init__(
        self,
        patterns: Union[list[str], dict[str, IntOrIntCountRange]],
        keep_matches: bool = True,
        mode: FilterModeType = "any",
        name: Optional[str] = None,
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize BasePatternsFilter.

        Parameters
        ----------
        patterns: Union[list[str], dict[str, IntOrIntCountRange]]
            List of patterns to allow in molecules.
            Alternatively, a dictionary can be passed with patterns as keys
            and an int for exact count or a tuple of minimum and maximum.
        keep_matches: bool, optional (default: True)
            If True, molecules containing the specified patterns are kept, else removed.
        mode: FilterModeType, optional (default: "any")
            If "any", at least one of the specified patterns must be present in the molecule.
            If "all", all of the specified patterns must be present in the molecule.
        name: Optional[str], optional (default: None)
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.
        """
        super().__init__(
            keep_matches=keep_matches, mode=mode, name=name, n_jobs=n_jobs, uuid=uuid
        )
        self.patterns = patterns  # type: ignore

    @property
    def patterns(self) -> dict[str, IntCountRange]:
        """Get allowed patterns as dict."""
        return self._patterns

    @patterns.setter
    def patterns(
        self,
        patterns: Union[list[str], dict[str, IntOrIntCountRange]],
    ) -> None:
        """Set allowed patterns as dict.

        Parameters
        ----------
        patterns: Union[list[str], dict[str, IntOrIntCountRange]]
            List of patterns.
        """
        if isinstance(patterns, (list, set)):
            self._patterns = {pat: (1, None) for pat in patterns}
        else:
            self._patterns = {
                pat: count_value_to_tuple(count) for pat, count in patterns.items()
            }
        self.patterns_mol_dict = list(self._patterns.keys())  # type: ignore

    @property
    def patterns_mol_dict(self) -> Mapping[str, RDKitMol]:
        """Get patterns as dict with RDKitMol objects."""
        return self._patterns_mol_dict

    @patterns_mol_dict.setter
    def patterns_mol_dict(self, patterns: list[str]) -> None:
        """Set patterns as dict with RDKitMol objects.

        Parameters
        ----------
        patterns: list[str]
            List of patterns.
        """
        self._patterns_mol_dict = {pat: self._pattern_to_mol(pat) for pat in patterns}

    @abc.abstractmethod
    def _pattern_to_mol(self, pattern: str) -> RDKitMol:
        """Convert pattern to Rdkitmol object.

        Parameters
        ----------
        pattern: str
            Pattern to convert.

        Returns
        -------
        RDKitMol
            RDKitMol object of the pattern.
        """

    @property
    def filter_elements(self) -> Mapping[str, IntCountRange]:
        """Get filter elements as dict."""
        return self.patterns

    def _calculate_single_element_value(
        self, filter_element: Any, value: RDKitMol
    ) -> int:
        """Calculate a single match count for a molecule.

        Parameters
        ----------
        filter_element: Any
            smarts to calculate match count for.
        value: RDKitMol
            Molecule to calculate smarts match count for.

        Returns
        -------
        int
            smarts match count value.
        """
        return len(value.GetSubstructMatches(self.patterns_mol_dict[filter_element]))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters of PatternFilter.

        Parameters
        ----------
        deep: bool, optional (default: True)
            If True, return the parameters of all subobjects that are PipelineElements.

        Returns
        -------
        dict[str, Any]
            Parameters of PatternFilter.
        """
        params = super().get_params(deep=deep)
        if deep:
            params["patterns"] = {
                pat: (count_tuple[0], count_tuple[1])
                for pat, count_tuple in self.patterns.items()
            }
        else:
            params["patterns"] = self.patterns
        return params

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters of PatternFilter.

        Parameters
        ----------
        parameters: Any
            Parameters to set.

        Returns
        -------
        Self
            Self.
        """
        parameter_copy = dict(parameters)
        if "patterns" in parameter_copy:
            self.patterns = parameter_copy.pop("patterns")
        super().set_params(**parameter_copy)
        return self
