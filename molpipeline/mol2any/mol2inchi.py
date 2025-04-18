"""Classes for transforming rdkit molecules to inchi."""

from __future__ import annotations

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.mol2any.mol2string import (
    MolToStringPipelineElement as _MolToStringPipelineElement,
)
from molpipeline.utils.molpipeline_types import RDKitMol


class MolToInchi(_MolToStringPipelineElement):
    """PipelineElement to transform a molecule to an INCHI string."""

    def pretransform_single(self, value: RDKitMol) -> str:
        """Transform a molecule to a INCHI-key string.

        Parameters
        ----------
        value: RDKitMol
            molecule to transform

        Returns
        -------
        str
            INCHI string
        """
        return str(Chem.MolToInchi(value))


class MolToInchiKey(_MolToStringPipelineElement):
    """PipelineElement to transform a molecule to an INCHI-Key string."""

    def __init__(
        self,
        name: str = "MolToInchiKey",
        n_jobs: int = 1,
        uuid: str | None = None,
    ):
        """Initialize MolToInchiKey.

        Parameters
        ----------
        name: str, default="MolToInchiKey"
            name of PipelineElement
        n_jobs: int, default=1
            number of jobs to use for parallelization
        uuid:  str | None, optional
            uuid of PipelineElement, by default None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> str:
        """Transform a molecule to an INCHI-key string.

        Parameters
        ----------
        value: RDKitMol
            molecule to transform

        Returns
        -------
        str
            INCHI-key of molecule.
        """
        return str(Chem.MolToInchiKey(value))
