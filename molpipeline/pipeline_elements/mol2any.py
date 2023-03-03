from typing import Any

from rdkit import Chem

from molpipeline.pipeline_elements.abstract_pipeline_elements import Mol2AnyPipe as _Mol2AnyPipe
from molpipeline.utils.molpipe_types import OptionalMol


class Mol2SmilesPipe(_Mol2AnyPipe):
    def __init__(self, name: str = "Mol2Smiles"):
        super(Mol2SmilesPipe, self).__init__(name)

    def fit(self, input_values: Any) -> None:
        pass

    def transform(self, mol_list: list[Chem.Mol]) -> list[str]:
        return [self.transform_single(mol) for mol in mol_list]

    def transform_single(self, mol: Chem.Mol) -> str:
        return str(Chem.MolToSmiles(mol))
