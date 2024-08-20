"""Abstract classes for filters."""

import abc
from typing import Any, Literal, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from molpipeline.abstract_pipeline_elements.core import MolToMolPipelineElement


class BasePatternsFilter(MolToMolPipelineElement, abc.ABC):
    """Filter to keep or remove molecules based on patterns."""

    def __init__(
        self,
        patterns: Union[
            list[str], dict[str, Union[int, tuple[Optional[int], Optional[int]]]]
        ],
        keep: bool = True,
        mode: Literal["any", "all"] = "any",
        name: Optional[str] = None,
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize BasePatternsFilter.

        Parameters
        ----------
        patterns: Union[list[str], dict[str, Union[int, tuple[Optional[int], Optional[int]]]]]
            List of patterns to allow in molecules.
            Alternatively, a dictionary can be passed with patterns as keys
            and an int for exact count or a tuple of minimum and maximum.
        keep: bool, optional (default: True)
            If True, molecules containing the specified patterns are kept, else removed.
        mode: Literal["any", "all"], optional (default: "any")
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
        self.patterns = patterns  # type: ignore
        self.keep = keep
        self.mode = mode

    @property
    def patterns(self) -> dict[str, tuple[Optional[int], Optional[int]]]:
        """Get allowed patterns as dict."""
        return self._patterns

    @patterns.setter
    def patterns(
        self,
        patterns: Union[
            list[str], dict[str, Union[int, tuple[Optional[int], Optional[int]]]]
        ],
    ) -> None:
        """Set allowed patterns as dict.

        Parameters
        ----------
        patterns: Union[list[str], dict[str, Union[int, tuple[Optional[int], Optional[int]]]]]
            List of patterns.
        """
        self._patterns: dict[str, tuple[Optional[int], Optional[int]]]
        if isinstance(patterns, (list, set)):
            self._patterns = {pat: (1, None) for pat in patterns}
        else:
            self._patterns = {}
            for pat, count in patterns.items():
                if isinstance(count, int):
                    self._patterns[pat] = (count, count)
                else:
                    self._patterns[pat] = count

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
        params["keep"] = self.keep
        params["mode"] = self.mode
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
        if "keep" in parameter_copy:
            self.keep = parameter_copy.pop("keep")
        if "mode" in parameter_copy:
            self.mode = parameter_copy.pop("mode")
        super().set_params(**parameter_copy)
        return self
