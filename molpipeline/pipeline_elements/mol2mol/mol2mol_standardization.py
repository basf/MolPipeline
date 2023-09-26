"""Classes for standardizing molecules."""
from __future__ import annotations

from typing import Any, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from rdkit import Chem
from rdkit.Chem import rdmolops, SanitizeMol  # pylint: disable=no-name-in-module
from rdkit.Chem import SaltRemover as rdkit_SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolHash

from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
    InvalidInstance,
)
from molpipeline.utils.molpipeline_types import OptionalMol, RDKitMol


MolHashing = Union[
    "rdMolHash.HashFunction.AnonymousGraph",
    "rdMolHash.HashFunction.ArthorSubstructureOrder",
    "rdMolHash.HashFunction.AtomBondCounts",
    "rdMolHash.HashFunction.CanonicalSmiles",
    "rdMolHash.HashFunction.DegreeVector",
    "rdMolHash.HashFunction.ElementGraph",
    "rdMolHash.HashFunction.ExtendedMurcko",
    "rdMolHash.HashFunction.HetAtomProtomer",
    "rdMolHash.HashFunction.HetAtomTautomer",
    "rdMolHash.HashFunction.HetAtomTautomerv2",
    "rdMolHash.HashFunction.Mesomer",
    "rdMolHash.HashFunction.MolFormula",
    "rdMolHash.HashFunction.MurckoScaffold",
    "rdMolHash.HashFunction.NetCharge",
    "rdMolHash.HashFunction.RedoxPair",
    "rdMolHash.HashFunction.Regioisomer",
    "rdMolHash.HashFunction.SmallWorldIndexBR",
    "rdMolHash.HashFunction.SmallWorldIndexBRL",
]


class CanonicalizeTautomerPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which canonicalizes tautomers of a molecule."""

    def __init__(
        self,
        name: str = "CanonicalizeTautomerPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize CanonicalizeTautomerPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Canonicalize tautomers of molecule.

        Parameters
        ----------
        value: RDKitMol
            Molecule to canonicalize tautomers from.

        Returns
        -------
        OptionalMol
            Canonicalized molecule if possible, else InvalidInstance.
        """
        enumerator = rdMolStandardize.TautomerEnumerator()
        return enumerator.Canonicalize(value)


class ChargeParentPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which returns charge-parent of a molecule, if possible."""

    def __init__(
        self,
        name: str = "ChargeParentPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize ChargeParentPipelineElement.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of jobs to use for parallelization
        uuid: Optional[str], optional
            uuid of PipelineElement, by default None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Return charge-parent of molecule, which is the largest fragment after neutralization.

        Parameters
        ----------
        value: RDKitMol
            Molecule to remove charges from.

        Returns
        -------
        OptionalMol
            Charge-parent of molecule if possible, else InvalidInstance.
        """
        return rdMolStandardize.ChargeParent(value)


class DeduplicateFragmentsByMolHashPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes duplicate fragments from a molecule.

    Duplicates are detected by comparing the MolHashes of the fragments.
    """

    hashing_method: MolHashing

    def __init__(
        self,
        name: str = "DeduplicateFragmentsByMolHashPipelineElement",
        hashing_method: MolHashing = rdMolHash.HashFunction.HetAtomTautomer,
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize DeduplicateFragmentsByMolHashPipelineElement.

        Parameters
        ----------
        name: str, optional (default: "DeduplicateFragmentsByInchiPipelineElement")
            Name of the pipeline element.
        hashing_method: MolHashing, optional (default: rdMolHash.HashFunction.HetAtomTautomer)
            MolHashing method to use for comparing fragments.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
            Unique identifier of the pipeline element.

        Returns
        -------
        None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.hashing_method = hashing_method

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Remove duplicate fragments from molecule.

        Parameters
        ----------
        value: RDKitMol
            Molecule to remove duplicate fragments from.

        Returns
        -------
        OptionalMol
            Molecule without duplicate fragments if possible, else InvalidInstance.
        """
        fragments = Chem.GetMolFrags(value, asMols=True)
        fragment_hash_list = [
            (rdMolHash.MolHash(fragment, self.hashing_method), fragment)
            for fragment in fragments
        ]
        if len(fragment_hash_list) == 0:
            return InvalidInstance(
                self.uuid, "Molecule contains no fragments.", self.name
            )
        unique_fragment_hashes = {fragment_hash_list[0][0]}
        recombined_fragment = fragment_hash_list[0][1]
        for fragment_hash, fragment in fragment_hash_list[1:]:
            if fragment_hash not in unique_fragment_hashes:
                unique_fragment_hashes.add(fragment_hash)
                recombined_fragment = Chem.CombineMols(recombined_fragment, fragment)

        for dict_key, dict_value in value.GetPropsAsDict(includeComputed=False).items():
            recombined_fragment.SetProp(dict_key, dict_value)
        return recombined_fragment


class LargestFragmentChooserPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which returns the largest fragment of a molecule."""

    def __init__(
        self,
        name: str = "LargestFragmentChooserPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize LargestFragmentChooserPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Return largest fragment of molecule.

        Parameters
        ----------
        value: RDKitMol
            Molecule to remove charges from.

        Returns
        -------
        OptionalMol
            Largest fragment of molecule if possible, else InvalidInstance.
        """
        return rdMolStandardize.LargestFragmentChooser().choose(value)


class MetalDisconnectorPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes bonds between organic compounds and metals."""

    def __init__(
        self,
        name: str = "MetalDisconnectorPipe",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MetalDisconnectorPipelineElement.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of jobs to use for parallelization
        uuid: Optional[str], optional
            uuid of PipelineElement, by default None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Cleave bonds with metals.

        Parameters
        ----------
        value: RDKitMol
            Molecule to disconnect metals from.

        Returns
        -------
        OptionalMol
            Molecule without bonds to metals if possible, else InvalidInstance.
        """
        mol = rdMolStandardize.MetalDisconnector().Disconnect(value)
        if mol is not None:
            # sometimes the molecule is not sanitized after disconnecting, e.g. RingInfo is not updated.
            SanitizeMol(mol)
        return mol


class RemoveStereoInformationPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes stereo-information from the molecule."""

    def __init__(
        self,
        name: str = "RemoveStereoInformationPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize RemoveStereoInformationPipelineElement.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of jobs to use for parallelization
        uuid: Optional[str], optional
            uuid of PipelineElement, by default None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Remove stereo-information in molecule.

        Parameters
        ----------
        value: RDKitMol
            Molecule to remove stereo-information from.

        Returns
        -------
        OptionalMol
            Molecule without stereo-information if possible, else InvalidInstance.
        """
        copy_mol = RDKitMol(value)
        rdmolops.RemoveStereochemistry(copy_mol)
        return copy_mol


class SaltRemoverPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes metal ions from molecule."""

    def __init__(
        self,
        name: str = "SaltRemoverPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize SaltRemoverPipe.

        Parameters
        ----------
        name: str
            Name of PipelineElement
        n_jobs: int
            Number of jobs to use for parallelization
        uuid: Optional[str], optional
            uuid of PipelineElement, by default None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Remove metal ions.

        Parameters
        ----------
        value: RDKitMol
            Molecule to remove metal ions from.

        Returns
        -------
        OptionalMol
            Molecule without metal ions if possible, else InvalidInstance.
        """
        salt_less_mol = rdkit_SaltRemover.SaltRemover().StripMol(value)
        return salt_less_mol


class SolventRemoverPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes defined fragments from a molecule."""

    _solvent_mol_list: list[RDKitMol]

    def __init__(
        self,
        solvent_smiles_list: Optional[list[str]] = None,
        name: str = "SolventRemoverPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize SolventRemoverPipelineElement.

        Taken from ChEMBL structure pipeline:
        https://github.com/chembl/ChEMBL_Structure_Pipeline/blob/master/chembl_structure_pipeline/data/solvents.smi

        Parameters
        ----------
        solvent_smiles_list: list[str], optional
            List of SMILES of fragments to remove, by default None which uses the default solvent list:
             - WATER	[OH2]
             - DICHLOROMETHANE	ClCCl
             - TRICHLOROMETHANE	ClC(Cl)Cl
             - ETHYL ACETATE	CCOC(=O)C
             - METHANOL	CO
             - PROPAN-2-OL	CC(C)O
             - ACETONE	CC(=O)C
             - DMSO	CS(=O)C
             - ETHANOL	CCO
        name: str, optional (default: "SolventRemoverPipelineElement")
            Name of the pipeline element.
        n_jobs: int, optional (default: 1)
            Number of parallel jobs to use.
        uuid: str, optional (default: None)
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        if solvent_smiles_list is None:
            solvent_smiles_list = [
                "[OH2]",
                "ClCCl",
                "ClC(Cl)Cl",
                "CCOC(=O)C",
                "CO",
                "CC(C)O",
                "CC(=O)C",
                "CS(=O)C",
                "CCO",
            ]
        self.solvent_smiles_list = solvent_smiles_list

    @property
    def solvent_mol_list(self) -> list[RDKitMol]:
        """Return molecule representation of smiles list."""
        return self._solvent_mol_list

    @property
    def solvent_smiles_list(self) -> list[str]:
        """Return the smiles list."""
        return self._solvent_smiles_list

    @solvent_smiles_list.setter
    def solvent_smiles_list(self, solvent_smiles_list: list[str]) -> None:
        """Set the smiles list."""
        self._solvent_smiles_list = solvent_smiles_list
        solvent_mol_list = []
        for smiles in solvent_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Could not convert {smiles} to a molecule.")
            solvent_mol_list.append(mol)
        self._solvent_mol_list = solvent_mol_list

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return parameters of pipeline element.

        Parameters
        ----------
        deep: bool, optional (default: True)
            If True, return a deep copy of the parameters, else only a shallow copy.

        Returns
        -------
        dict[str, Any]
            Parameters of pipeline element.
        """
        params = super().get_params(deep=deep)
        if deep:
            params["solvent_smiles_list"] = [
                str(smi) for smi in self.solvent_smiles_list
            ]
        else:
            params["solvent_smiles_list"] = self.solvent_smiles_list
        return params

    def set_params(self, parameters: dict[str, Any]) -> Self:
        """Set parameters of pipeline element.

        Parameters
        ----------
        parameters: dict[str, Any]
            Parameters to set.

        Returns
        -------
        Self
            Pipeline element with set parameters.
        """
        param_copy = dict(parameters)
        solvent_smiles_list = param_copy.pop("solvent_smiles_list", None)
        if solvent_smiles_list is not None:
            self.solvent_smiles_list = solvent_smiles_list
        super().set_params(param_copy)
        return self

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Remove all fragments from molecule.

        Parameters
        ----------
        value: RDKitMol
            Molecule to remove fragments from.

        Returns
        -------
        OptionalMol
            Molecule without fragments if possible, else InvalidInstance.
        """
        kept_fragments = []
        for fragment in Chem.GetMolFrags(value, asMols=True):
            n_atoms_f = fragment.GetNumAtoms()
            n_bonds_f = fragment.GetNumBonds()
            for solvent_mol in self._solvent_mol_list:
                if (
                    n_atoms_f == solvent_mol.GetNumAtoms()
                    and n_bonds_f == solvent_mol.GetNumBonds()
                    and fragment.HasSubstructMatch(solvent_mol)
                ):
                    break  # if it matches, stop searching
            else:  # no break: no match
                kept_fragments.append(fragment)
        if len(kept_fragments) == 0:
            return InvalidInstance(self.uuid, "All fragments were removed.", self.name)
        combined_fragments = kept_fragments[0]
        for fragment in kept_fragments[1:]:
            combined_fragments = Chem.CombineMols(combined_fragments, fragment)
        for dict_key, dict_value in value.GetPropsAsDict(includeComputed=False).items():
            combined_fragments.SetProp(dict_key, dict_value)
        return combined_fragments


class UnchargePipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes charges in a molecule, if possible."""

    def __init__(
        self,
        name: str = "UnchargePipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize UnchargePipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    def pretransform_single(self, value: RDKitMol) -> OptionalMol:
        """Remove charges of molecule.

        Parameters
        ----------
        value: RDKitMol
            Molecule to remove charges from.

        Returns
        -------
        OptionalMol
            Uncharged molecule if possible, else InvalidInstance.
        """
        return rdMolStandardize.Uncharger().uncharge(value)
