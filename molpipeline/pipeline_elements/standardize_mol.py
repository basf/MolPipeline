from typing import Any

from rdkit import Chem
from rdkit.Chem import SaltRemover as rdkit_SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from molpipeline.pipeline_elements.abstract_pipeline_elements import Mol2MolPipe as _Mol2MolPipe
from molpipeline.utils.molpipe_types import OptionalMol


class RemoveChargePipe(_Mol2MolPipe):
    def __init__(self, name: str = "RemoveChargePipe") -> None:
        super(RemoveChargePipe, self).__init__(name)

    def fit(self, input_values: Any) -> None:
        pass

    def transform_single(self, mol: Chem.Mol) -> OptionalMol:
        return rdMolStandardize.ChargeParent(mol)


class MetalDisconnectorPipe(_Mol2MolPipe):
    def __init__(self, name: str = "MetalDisconnectorPipe") -> None:
        super(MetalDisconnectorPipe, self).__init__(name)
        self._metal_disconnector = rdMolStandardize.MetalDisconnector()

    def fit(self, input_values: Any) -> None:
        pass

    def transform_single(self, mol: Chem.Mol) -> OptionalMol:
        return self._metal_disconnector.Disconnect(mol)


class SaltRemoverPipe(_Mol2MolPipe):
    def __init__(self, name: str = "SaltRemoverPipe") -> None:
        super(SaltRemoverPipe, self).__init__(name)
        self._salt_remover = rdkit_SaltRemover.SaltRemover()

    def fit(self, input_values: Any) -> None:
        pass

    def transform_single(self, mol: Chem.Mol) -> OptionalMol:
        return self._salt_remover.StripMol(mol)
