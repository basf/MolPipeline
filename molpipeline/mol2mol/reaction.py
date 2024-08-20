"""Pipline elements for mol to mol transformations."""

# pylint: disable=too-many-arguments

from __future__ import annotations

from typing import Any, Literal, Optional

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import copy
import warnings

from rdkit.Chem import AllChem

from molpipeline.abstract_pipeline_elements.core import (
    InvalidInstance,
    MolToMolPipelineElement,
)
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol


class MolToMolReaction(MolToMolPipelineElement):
    """PipelineElement which transforms the input according to the specified reaction."""

    additive_list: list[RDKitMol]
    handle_multi: Literal["pass", "warn", "raise"]
    _reaction: AllChem.ChemicalReaction

    def __init__(
        self,
        reaction: AllChem.ChemicalReaction,
        additive_list: list[RDKitMol],
        handle_multi: Literal["pass", "warn", "raise"] = "warn",
        name: str = "MolToMolReaction",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToMolReaction.

        Parameters
        ----------
        reaction: AllChem.ChemicalReaction
            Reaction which is applied to input.
        additive_list: list[Chem.Mol]
            Molecules which are added as educts to the reaction, but are not part of input.
        handle_multi: Literal["pass", "warn", "raise"]
            How to handle reaction where multiple products are possible.
        name: str, optional (default="MolToMolReaction")
            Name of PipelineElement.
        n_jobs: int, optional (default=1)
            Number of cores used.
        uuid: str | None, optional (default=None)
            UUID of the pipeline element. If None, a random UUID is generated.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
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

    def set_params(self, **parameters: Any) -> Self:
        """Set the parameters.

        Parameters
        ----------
        parameters: Any
            Dictionary containing parameters to be set.

        Returns
        -------
        Self
            MolToMolReaction with updated parameters.
        """
        super().set_params(**parameters)
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

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Apply reaction to molecule.

        Parameters
        ----------
        value: RDKitMol
            Molecule to apply reaction to.

        Returns
        -------
        OptionalMol
            Product of reaction if possible, else InvalidInstance.
        """
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
            return InvalidInstance(
                self.uuid, "Reaction did not yield any product.", self.name
            )
        product = product_list[0][0]
        AllChem.SanitizeMol(product)
        return product
