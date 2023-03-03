from typing import Any

from rdkit import Chem

from molpipeline.pipeline_elements.abstract_pipeline_elements import Any2Mol
from molpipeline.utils.molpipe_types import OptionalMol


class Smiles2Mol(Any2Mol):
    def __init__(self, identifier: str = "smiles", name: str = "smiles2Mol") -> None:
        self.identifier = identifier
        super().__init__(name)

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


class SDF2Mol(Any2Mol):
    identifier: str
    mol_counter: int

    def __init__(self, identifier: str = "enumerate", name: str = "SDF2Mol") -> None:
        super().__init__(name)
        self.identifier = identifier
        self.mol_counter = 0

    def finish(self) -> None:
        self.mol_counter = 0

    def fit(self, input_values: Any) -> None:
        pass

    def transform(self, input_values: str) -> list[OptionalMol]:
        molecule_list = [self.transform_single(sdf_str) for sdf_str in input_values]
        self.finish()
        return molecule_list

    def transform_single(self, input_value: Chem.Mol) -> OptionalMol:
        mol = Chem.MolFromMolBlock(input_value)
        if self.identifier == "smiles":
            mol.SetProp("identifier", self.mol_counter)
        self.mol_counter += 1
        return mol
