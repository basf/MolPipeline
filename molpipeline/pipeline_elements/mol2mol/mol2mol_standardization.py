"""Classes for standardizing molecules."""
from __future__ import annotations
from typing import Any

from rdkit import Chem
from rdkit.Chem import SaltRemover as rdkit_SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from molpipeline.abstract_pipeline_elements.core import (
    NONE_HANDLING_OPTIONS,
    MolToMolPipelineElement as _MolToMolPipelineElement,
)
from molpipeline.utils.molpipe_types import OptionalMol


class RemoveChargePipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes charges in a molecule, if possible."""

    def __init__(
        self,
        none_handling: NONE_HANDLING_OPTIONS = "raise",
        fill_value: Any = None,
        name: str = "RemoveChargePipe",
        n_jobs: int = 1,
    ) -> None:
        """Initialize RemoveChargePipelineElement."""
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )

    @property
    def params(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        return {"name": self.name, "n_jobs": self.n_jobs}

    def copy(self) -> RemoveChargePipelineElement:
        """Create a copy of the object."""
        return RemoveChargePipelineElement(**self.params)

    def _transform_single(self, value: Chem.Mol) -> OptionalMol:
        """Remove charges of molecule."""
        return rdMolStandardize.ChargeParent(value)


class MetalDisconnectorPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes bonds between organic compounds and metals."""

    def __init__(
        self,
        none_handling: NONE_HANDLING_OPTIONS = "raise",
        fill_value: Any = None,
        name: str = "MetalDisconnectorPipe",
        n_jobs: int = 1,
    ) -> None:
        """Initialize MetalDisconnectorPipelineElement."""
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )
        self._metal_disconnector = rdMolStandardize.MetalDisconnector()

    def _transform_single(self, value: Chem.Mol) -> OptionalMol:
        """Cleave bonds with metals."""
        return self._metal_disconnector.Disconnect(value)

    @property
    def params(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        return {"name": self.name, "n_jobs": self.n_jobs}

    def copy(self) -> MetalDisconnectorPipelineElement:
        """Create a copy of the object."""
        return MetalDisconnectorPipelineElement(**self.params)


class SaltRemoverPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes metal ions from molecule."""

    def __init__(
        self,
        none_handling: NONE_HANDLING_OPTIONS = "raise",
        fill_value: Any = None,
        name: str = "SaltRemoverPipelineElement",
        n_jobs: int = 1,
    ) -> None:
        """Initialize SaltRemoverPipe."""
        super().__init__(
            none_handling=none_handling, fill_value=fill_value, name=name, n_jobs=n_jobs
        )
        self._salt_remover = rdkit_SaltRemover.SaltRemover()

    @property
    def params(self) -> dict[str, Any]:
        """Return all parameters defining the object."""
        return {"name": self.name, "n_jobs": self.n_jobs}

    def copy(self) -> SaltRemoverPipelineElement:
        """Create a copy of the object."""
        return SaltRemoverPipelineElement(**self.params)

    def _transform_single(self, value: Chem.Mol) -> OptionalMol:
        """Remove metal ions."""
        return self._salt_remover.StripMol(value)
