"""Classes for standardizing molecules."""
from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]
from rdkit.Chem import rdmolops, SanitizeMol
from rdkit.Chem import SaltRemover as rdkit_SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from molpipeline.abstract_pipeline_elements.core import (
    MolToMolPipelineElement as _MolToMolPipelineElement,
    InvalidInstance,
)
from molpipeline.utils.molpipeline_types import OptionalMol


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


class DeduplicateFragmentsByInchiPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes duplicate fragments from a molecule.

    Duplicates are detected by comparing the InChI of the fragments.
    """

    def __int__(
        self,
        name: str = "DeduplicateFragmentsByInchiPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize DeduplicateFragmentsByInchiPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

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
        fragment_inchis = [Chem.MolToInchi(fragment) for fragment in fragments]
        unique_fragment_list = sorted(set(fragment_inchis))
        unique_fragments = [Chem.MolFromInchi(inchi) for inchi in unique_fragment_list]
        if None in unique_fragments:
            return InvalidInstance(self.uuid, "Could not recreate molecule from InChI.")
        if len(unique_fragments) == 1:
            combined_fragments = unique_fragments[0]
        else:
            combined_fragments = Chem.CombineMols(*unique_fragments)
        for key, value in value.GetPropsAsDict(includeComputed=False).items():
            combined_fragments.SetProp(key, value)
        return combined_fragments


class DeduplicateFragmentsBySmilesPipelineElement(_MolToMolPipelineElement):
    """MolToMolPipelineElement which removes duplicate fragments from a molecule.

    Duplicates are detected by comparing the SMILES of the fragments.
    """

    def __int__(
        self,
        name: str = "DeduplicateFragmentsBySmilesPipelineElement",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize DeduplicateFragmentsBySmilesPipelineElement."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

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
        fragment_smiles = [Chem.MolToSmiles(fragment) for fragment in fragments]
        unique_fragment_list = sorted(set(fragment_smiles))
        unique_fragments = [
            Chem.MolFromSmiles(smiles) for smiles in unique_fragment_list
        ]
        if None in unique_fragments:
            return InvalidInstance(
                self.uuid, "Could not recreate molecule from SMILES."
            )
        if len(unique_fragments) == 1:
            combined_fragments = unique_fragments[0]
        else:
            combined_fragments = Chem.CombineMols(*unique_fragments)
        for key, value in value.GetPropsAsDict(includeComputed=False).items():
            combined_fragments.SetProp(key, value)
        return combined_fragments


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
