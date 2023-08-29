"""Classes for standardizing molecules."""
from __future__ import annotations

from typing import Optional

from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]
from rdkit.Chem import rdmolops, SanitizeMol
from rdkit.Chem import SaltRemover as rdkit_SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
)
from molpipeline.utils.molpipeline_types import OptionalMol


class MetalDisconnectorPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes bonds between organic compounds and metals."""

    def __init__(
        self,
        name: str = "MetalDisconnectorPipe",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MetalDisconnectorPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Cleave bonds with metals."""
        mol = rdMolStandardize.MetalDisconnector().Disconnect(value)
        if mol is not None:
            # sometimes the molecule is not sanitized after disconnecting, e.g. RingInfo is not updated.
            SanitizeMol(mol)
        return mol


class RemoveChargePipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which returns charge-parent of a molecule, if possible."""

    def __init__(
        self,
        name: str = "RemoveChargePipe",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize RemoveChargePipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Remove charges of molecule."""
        return rdMolStandardize.ChargeParent(value)


class RemoveStereoInformationPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes stereo-information from the molecule."""

    def __init__(
        self,
        name: str = "RemoveStereoInformationPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize RemoveStereoInformationPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Remove stereo-information in molecule."""
        copy_mol = RDKitMol(value)
        rdmolops.RemoveStereochemistry(copy_mol)
        return copy_mol


class SaltRemoverPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes metal ions from molecule."""

    def __init__(
        self,
        name: str = "SaltRemoverPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize SaltRemoverPipe."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Remove metal ions."""
        salt_less_mol = rdkit_SaltRemover.SaltRemover().StripMol(value)
        return salt_less_mol


class UnchargePipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes charges in a molecule, if possible."""

    def __init__(
        self,
        name: str = "UnchargePipe",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize UnchargePipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Remove charges of molecule."""
        return rdMolStandardize.Uncharger().uncharge(value)
