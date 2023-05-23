"""Pipline elements for mol to mol transformations."""
# pylint: disable=too-many-arguments

from __future__ import annotations

from typing import Any, Literal

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self
import warnings

import copy
from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]
from rdkit.Chem import AllChem

from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement,
    NoneHandlingOptions,
)
from molpipeline.utils.molpipeline_types import OptionalMol


class MolToMolReactionPipelineElement(MolToMolPipelineElement):
    """PipelineElement which transforms the input according to the specified reaction."""

    additive_list: list[RDKitMol]
    handle_multi: Literal["pass", "warn", "raise"]
    _reaction: AllChem.ChemicalReaction

    def __init__(
        self,
        reaction: AllChem.ChemicalReaction,
        additive_list: list[RDKitMol],
        handle_multi: Literal["pass", "warn", "raise"] = "warn",
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = None,
        name: str = "MolToMolReactionPipelineElement",
        n_jobs: int = 1,
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
        super().__init__(
            name=name, n_jobs=n_jobs, none_handling=none_handling, fill_value=fill_value
        )
        self.reaction = reaction
        self.additive_list = additive_list
        self.handle_multi = handle_multi

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all parameters defining the object.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters defining the object.
        """
        parameters = super().get_params(deep)
        if deep:
            parameters["reaction"] = AllChem.ChemicalReaction(self.reaction)
            parameters["additive_list"] = [
                RDKitMol(additive) for additive in self.additive_list
            ]
            parameters["handle_multi"] = copy.copy(self.handle_multi)
        else:
            parameters["reaction"] = self.reaction
            parameters["additive_list"] = self.additive_list
            parameters["handle_multi"] = self.handle_multi
        return parameters

    def set_params(self, parameters: dict[str, Any]) -> Self:
        """Set the parameters.

        Parameters
        ----------
        parameters: dict[str, Any]
            Dictionary containing parameters to be set.

        Returns
        -------
        Self
            MolToMolReactionPipelineElement with updated parameters.
        """
        super().set_params(parameters)
        if "reaction" in parameters:
            self.reaction = parameters["reaction"]
        if "additive_list" in parameters:
            self.additive_list = parameters["additive_list"]
        if "handle_multi" in parameters:
            self.handle_multi = parameters["handle_multi"]
        return self

    @property
    def reaction(self) -> AllChem.ChemicalReaction:
        """Get the reaction which is applied to the input molecule."""
        return self._reaction

    @reaction.setter
    def reaction(self, reaction: AllChem.ChemicalReaction) -> None:
        """Set the reaction which is applied to the input molecule.

        Parameters
        ----------
        reaction: AllChem.ChemicalReaction
            Reaction which is applied to molecules.
        Returns
        -------
        None
        """
        if not isinstance(reaction, AllChem.ChemicalReaction):
            raise TypeError("Not a Chemical reaction!")
        self._reaction = reaction

    def _transform_single(self, value: RDKitMol) -> OptionalMol:
        """Apply reaction to molecule."""
        mol = value  # Only value to keep signature consistent.
        reactant_list: list[RDKitMol] = list(self.additive_list)
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
