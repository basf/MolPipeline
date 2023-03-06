"""Classes for standardizing molecules."""

from rdkit import Chem
from rdkit.Chem import SaltRemover as rdkit_SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
)
from molpipeline.utils.molpipe_types import OptionalMol


class RemoveChargePipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes charges in a molecule, if possible."""

    def __init__(self, name: str = "RemoveChargePipe") -> None:
        """Initialize RemoveChargePipelineElement."""
        super().__init__(name)

    def _transform_single(self, value: Chem.Mol) -> OptionalMol:
        """Remove charges of molecule."""
        return rdMolStandardize.ChargeParent(value)


class MetalDisconnectorPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes bonds between organic compounds and metals."""

    def __init__(self, name: str = "MetalDisconnectorPipe") -> None:
        """Initialize MetalDisconnectorPipelineElement."""
        super().__init__(name)
        self._metal_disconnector = rdMolStandardize.MetalDisconnector()

    def _transform_single(self, value: Chem.Mol) -> OptionalMol:
        """Cleave bonds with metals."""
        return self._metal_disconnector.Disconnect(value)


class SaltRemoverPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes metal ions from molecule."""

    def __init__(self, name: str = "SaltRemoverPipelineElement") -> None:
        """Initialize SaltRemoverPipe."""
        super().__init__(name)
        self._salt_remover = rdkit_SaltRemover.SaltRemover()

    def _transform_single(self, value: Chem.Mol) -> OptionalMol:
        """Remove metal ions."""
        return self._salt_remover.StripMol(value)
