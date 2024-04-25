"""Init the module for mol2any pipeline elements."""

from molpipeline.mol2any.mol2bin import MolToBinary
from molpipeline.mol2any.mol2concatinated_vector import MolToConcatenatedVector
from molpipeline.mol2any.mol2inchi import MolToInchi, MolToInchiKey
from molpipeline.mol2any.mol2morgan_fingerprint import MolToMorganFP
from molpipeline.mol2any.mol2net_charge import MolToNetCharge
from molpipeline.mol2any.mol2rdkit_phys_chem import MolToRDKitPhysChem
from molpipeline.mol2any.mol2smiles import MolToSmiles

__all__ = [
    "MolToBinary",
    "MolToConcatenatedVector",
    "MolToSmiles",
    "MolToMorganFP",
    "MolToNetCharge",
    "MolToInchi",
    "MolToInchiKey",
    "MolToRDKitPhysChem",
]

try:
    from molpipeline.mol2any.mol2chemprop import MolToChemprop  # noqa

    __all__.append("MolToChemprop")
except ImportError:
    pass
