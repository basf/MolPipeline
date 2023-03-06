"""Pipline elements for mol to mol transformations."""
from typing import Literal
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem


from molpipeline.abstract_pipeline_elements.core import MolToMolPipelineElement
from molpipeline.utils.molpipe_types import OptionalMol


class MolToMolReactionPipelineElement(MolToMolPipelineElement):
    """PipelineElement which transforms the input according to the specified reaction."""

    additive_list: list[Chem.Mol]
    handle_multi: Literal["pass", "warn", "raise"]
    _reaction: AllChem.ChemicalReaction

    def __init__(
        self,
        reaction: AllChem.ChemicalReaction,
        additive_list: list[Chem.Mol],
        handle_multi: Literal["pass", "warn", "raise"] = "warn",
        name: str = "MolToMolReactionPipelineElement",
    ) -> None:
        """Initialize MolToMolReactionPipelineElement.

        Parameters
        ----------
        reaction: AllChem.ChemicalReaction
            Reaction which is applied to input.
        additive_list: list[Chem.Mol]
            Molecules which are added as educts to the reaction, but are not part of input.
        handle_multi: Literal["pass", "warn", "raise"]
            How to handle reaction where multiple products are possible.
        name: str
            Name of PipelineElement
        """
        super().__init__(name)
        self.reaction = reaction
        self.additive_list = additive_list
        self.handle_multi = handle_multi

    @property
    def reaction(self) -> AllChem.ChemicalReaction:
        """Get the reaction which is applied to the input molecule."""
        return self._reaction

    @reaction.setter
    def reaction(self, reaction: AllChem.ChemicalReaction) -> None:
        """Set the reaction which is applied to the input molecule."""
        if not isinstance(reaction, AllChem.ChemicalReaction):
            raise TypeError("Not a Chemical reaction!")
        self._reaction = reaction

    def transform_single(self, value: Chem.Mol) -> OptionalMol:
        """Apply reaction to molecule."""
        mol = value  # Only value to keep signature consistent.
        reactant_list: list[Chem.Mol] = list(self.additive_list)
        reactant_list.append(mol)
        product_list = self.reaction.RunReactants(reactant_list)

        if len(product_list) > 1:
            if self.handle_multi == "warn":
                warnings.warn(
                    "Not able to handle multiple reactions. An arbitrary reaction is selected."
                )
            elif self.handle_multi == "raise":
                if mol.HasProp("identifier"):
                    mol_id = mol.GetProp("identifier")
                else:
                    mol_id = "None"
                raise ValueError(
                    f"Not able to handle multiple reactions: Mol ID: {mol_id}"
                )

        if len(product_list) == 0:
            return None
        product = product_list[0][0]
        AllChem.SanitizeMol(product)
        return product
