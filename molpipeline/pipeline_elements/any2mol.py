from rdkit import Chem

from molpipeline.pipeline_elements.abstract_pipeline_elements import Any2Mol
from molpipeline.utils.molpipe_types import OptionalMol


class Smiles2Mol(Any2Mol):
    def __init__(self, identifier: str = "smiles"):
        self.identifier = identifier

    def fit(self, input_values: list[str]) -> None:
        pass

    def transform(self, smiles_list: list[str]) -> list[OptionalMol]:
        return [self.transform_single(smiles) for smiles in smiles_list]

    def transform_single(self, smiles: str) -> OptionalMol:
        mol: Chem.Mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        if self.identifier == "smiles":
            mol.SetProp("identifier", smiles)
        return mol
