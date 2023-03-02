from __future__ import annotations
from rdkit import Chem


class AtomEnvironment:
    """ "A Class to store environment-information for fingerprint features"""

    def __init__(self, environment_atoms: set[int]):
        self.environment_atoms = environment_atoms  # set of all atoms within radius


class CircularAtomEnvironment(AtomEnvironment):
    """ "A Class to store environment-information for morgan-fingerprint features"""
    def __init__(self, central_atom: int, radius: int, environment_atoms: set[int]):
        super().__init__(environment_atoms)
        self.central_atom = central_atom
        self.radius = radius

    @classmethod
    def from_mol(
            cls, mol: Chem.Mol, central_atom_index: int, radius: int) -> CircularAtomEnvironment:
        """ Generate class from mol, using location (central_atom_index) and the radius.

        Parameters
        ----------
        mol: Chem.Mol
            Molecule from which the environment is derived.
        central_atom_index: int
            Index of central atom in fingerprint.
        radius: int
            Radius of feature.

        Returns
        -------
        CircularAtomEnvironment
        """

        if radius == 0:
            return CircularAtomEnvironment(central_atom_index, radius, {central_atom_index})
        else:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, central_atom_index)
            amap: dict[int, int] = dict()
            _ = Chem.PathToSubmol(mol, env, atomMap=amap)
            env_atoms = amap.keys()
            return CircularAtomEnvironment(central_atom_index, radius, set(env_atoms))


def bit2atom_mapping(
        mol_obj: Chem.Mol,
        bit_dict: dict[int, list[tuple[int, int]]]
) -> dict[int, list[CircularAtomEnvironment]]:

    result_dict: dict[int, list[CircularAtomEnvironment]] = dict()
    # Iterating over all present bits and respective matches
    for bit, matches in bit_dict.items():  # type: int, list[tuple[int, int]]
        result_dict[bit] = []
        for central_atom, radius in matches:  # type: int, int
            bit_env = CircularAtomEnvironment.from_mol(mol_obj, central_atom, radius)
            result_dict[bit].append(bit_env)

    return result_dict
