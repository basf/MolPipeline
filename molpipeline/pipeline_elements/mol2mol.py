"""Pipline elements for mol to mol transformations."""

from rdkit import Chem
from rdkit.Chem import AllChem

from molpipeline.pipeline_elements.abstract_pipeline_elements import Mol2MolPipe
from molpipeline.utils.molpipe_types import OptionalMol


class ReactionPipe(Mol2MolPipe):
    def __init__(self, reaction: AllChem.ChemicalReaction, additive_list: list[Chem.Mol]):
        self.reaction = reaction
        self.additive_list = additive_list

    @property
    def reaction(self) -> AllChem.ChemicalReaction:
        return self._reaction

    @reaction.setter
    def reaction(self, reaction: AllChem.ChemicalReaction) -> None:
        if not isinstance(reaction, AllChem.ChemicalReaction):
            raise TypeError("Not a Chemical reaction!")
        self._reaction = reaction

    def transform_mol(self, mol: Chem.Mol) -> OptionalMol:
        reactant_list: list[Chem.Mol] = list(self.additive_list)
        reactant_list.append(mol)
        product_list = self.reaction.RunReactants(reactant_list)
        # TODO: handle multiple reaction matches
        # TODO: handle multiple products
        if len(product_list) > 1:
            pass
            #warnings.warn("Not able to handle multiple reactions. An arbriraty reaction is selected.")
        if len(product_list) == 0:
            return None
        product = product_list[0][0]
        AllChem.SanitizeMol(product)
        return product
