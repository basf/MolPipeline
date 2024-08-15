"""Pipeline elements for converting instances to bool."""

from typing import Any

from molpipeline.abstract_pipeline_elements.core import (
    InvalidInstance,
    MolToAnyPipelineElement,
)


class MolToBool(MolToAnyPipelineElement):
    """Element to generate a bool array from input."""

    def __init__(
        self,
        name: str = "Mol2Bool",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize MolToBinaryPipelineElement.

        Parameters
        ----------
        name: str, optional (default="Mol2Bool")
            name of PipelineElement
        n_jobs: int, optional (default=1)
            number of jobs to use for parallelization
        uuid: Optional[str], optional (default=None)
            uuid of PipelineElement, by default None

        Returns
        -------
        None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: Any) -> bool:
        """Transform a value to a bool representation.

        Parameters
        ----------
        value: Any
            Value to be transformed to bool representation.

        Returns
        -------
        str
            Binary representation of molecule.
        """
        if isinstance(value, InvalidInstance):
            return False
        return True
