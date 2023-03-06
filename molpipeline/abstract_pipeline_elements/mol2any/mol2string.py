import abc

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement


class MolToStringPipelineElement(MolToAnyPipelineElement, abc.ABC):
    """Abstract class for PipelineElements which transform molecules to integer vectors."""
    _output_type = str

    def transform(self, value_list: list[Chem.Mol]) -> list[str]:
        """Transform the list of molecules to sparse matrix."""
        return super().transform(value_list)

    @abc.abstractmethod
    def _transform_single(self, value: Chem.Mol) -> str:
        """Transform mol to a string.

        Parameters
        ----------
        value: Chem.Mol

        Returns
        -------
        str
        """