"""Tests for SDF to mol."""

import unittest

from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.any2mol.sdf2mol import SDFToMol


class TestSDFToMol(unittest.TestCase):
    """Test class for SDFToMol."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sdf_str_benzaldehyde = """240
  -OEChem-05082503512D

 14 14  0     0  0  0  0  0  0999 V2000
    3.7321    1.7500    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.8660    0.2500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000   -0.2500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.7321   -0.2500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000   -1.2500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.7321   -1.2500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.8660   -1.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.8660    1.2500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4631    0.0600    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.2690    0.0600    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4631   -1.5600    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.2690   -1.5600    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.8660   -2.3700    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.3291    1.5600    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  8  2  0  0  0  0
  2  3  2  0  0  0  0
  2  4  1  0  0  0  0
  2  8  1  0  0  0  0
  3  5  1  0  0  0  0
  3  9  1  0  0  0  0
  4  6  2  0  0  0  0
  4 10  1  0  0  0  0
  5  7  2  0  0  0  0
  5 11  1  0  0  0  0
  6  7  1  0  0  0  0
  6 12  1  0  0  0  0
  7 13  1  0  0  0  0
  8 14  1  0  0  0  0
M  END
> <PUBCHEM_COMPOUND_CID>
240

> <PUBCHEM_COMPOUND_CANONICALIZED>
1

> <PUBCHEM_CACTVS_COMPLEXITY>
72.5

$$$$
"""

        self.sdf_str_aspirin = """2244
  -OEChem-05082504263D

 21 21  0     0  0  0  0  0  0999 V2000
    1.2333    0.5540    0.7792 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6952   -2.7148   -0.7502 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.7958   -2.1843    0.8685 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.7813    0.8105   -1.4821 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0857    0.6088    0.4403 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7927   -0.5515    0.1244 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7288    1.8464    0.4133 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1426   -0.4741   -0.2184 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0787    1.9238    0.0706 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7855    0.7636   -0.2453 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1409   -1.8536    0.1477 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1094    0.6715   -0.3113 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.5305    0.5996    0.1635 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1851    2.7545    0.6593 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7247   -1.3605   -0.4564 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5797    2.8872    0.0506 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.8374    0.8238   -0.5090 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.7290    1.4184    0.8593 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.2045    0.6969   -0.6924 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.7105   -0.3659    0.6426 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2555   -3.5916   -0.7337 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  5  1  0  0  0  0
  1 12  1  0  0  0  0
  2 11  1  0  0  0  0
  2 21  1  0  0  0  0
  3 11  2  0  0  0  0
  4 12  2  0  0  0  0
  5  6  1  0  0  0  0
  5  7  2  0  0  0  0
  6  8  2  0  0  0  0
  6 11  1  0  0  0  0
  7  9  1  0  0  0  0
  7 14  1  0  0  0  0
  8 10  1  0  0  0  0
  8 15  1  0  0  0  0
  9 10  2  0  0  0  0
  9 16  1  0  0  0  0
 10 17  1  0  0  0  0
 12 13  1  0  0  0  0
 13 18  1  0  0  0  0
 13 19  1  0  0  0  0
 13 20  1  0  0  0  0
M  END
> <PUBCHEM_COMPOUND_CID>
2244

> <PUBCHEM_CONFORMER_RMSD>
0.6

$$$$
"""
        self.sdf_str_benzaldehyde_aspirin = (
            self.sdf_str_benzaldehyde + self.sdf_str_aspirin
        )
        self.invalid_sdf = "Not an SDF string"

    def test_initialization(self) -> None:
        """Test initialization of SDFToMol."""
        sdf2mol = SDFToMol(identifier="smiles", name="CustomName", n_jobs=2)
        self.assertEqual(sdf2mol.identifier, "smiles")
        self.assertEqual(sdf2mol.name, "CustomName")
        self.assertEqual(sdf2mol.n_jobs, 2)
        self.assertEqual(sdf2mol.mol_counter, 0)

    def test_pretransform_valid_sdf(self) -> None:
        """Test transformation of valid SDF string.

        Raises
        ------
        AssertionError
            If the transformation does not return a valid molecule.

        """
        sdf2mol = SDFToMol()
        # test sdf string with single molecule: benzaldehyde
        mol = sdf2mol.pretransform_single(self.sdf_str_benzaldehyde)
        self.assertIsInstance(mol, Chem.Mol)
        if not isinstance(mol, Chem.Mol):
            # necessary for mypy
            raise AssertionError("Expected a Chem.Mol object")
        self.assertEqual(mol.GetNumAtoms(), 8)

        # test sdf string with single molecule: aspirin
        mol = sdf2mol.pretransform_single(self.sdf_str_aspirin)
        self.assertIsInstance(mol, Chem.Mol)
        if not isinstance(mol, Chem.Mol):
            # necessary for mypy
            raise AssertionError("Expected a Chem.Mol object")
        self.assertEqual(mol.GetNumAtoms(), 13)

        # test sdf string with multiple molecules: benzaldehyde and aspirin
        # the current behavior is to return the first molecule only
        mol = sdf2mol.pretransform_single(self.sdf_str_benzaldehyde_aspirin)
        self.assertIsInstance(mol, Chem.Mol)
        if not isinstance(mol, Chem.Mol):
            # necessary for mypy
            raise AssertionError("Expected a Chem.Mol object")
        self.assertEqual(mol.GetNumAtoms(), 8)

    def test_pretransform_invalid_sdf(self) -> None:
        """Test handling of invalid SDF input."""
        result = SDFToMol().pretransform_single(self.invalid_sdf)
        self.assertIsInstance(result, InvalidInstance)

    def test_transform(self) -> None:
        """Test transform function.

        Raises
        ------
        AssertionError
            If the transformation does not return a valid molecule.

        """
        # test list of sdf strings
        mols = SDFToMol().transform([self.sdf_str_benzaldehyde, self.sdf_str_aspirin])
        self.assertEqual(len(mols), 2)
        self.assertIsInstance(mols[0], Chem.Mol)
        self.assertIsInstance(mols[1], Chem.Mol)
        if not isinstance(mols[0], Chem.Mol):
            # necessary for mypy
            raise AssertionError("Expected a Chem.Mol object")
        self.assertEqual(mols[0].GetNumAtoms(), 8)
        if not isinstance(mols[1], Chem.Mol):
            # necessary for mypy
            raise AssertionError("Expected a Chem.Mol object")
        self.assertEqual(mols[1].GetNumAtoms(), 13)

        # test single sdf string with multiple molecules
        # only the first molecule in the SDF string will be read
        mols = SDFToMol().transform([self.sdf_str_benzaldehyde_aspirin])
        self.assertEqual(len(mols), 1)
        self.assertIsInstance(mols[0], Chem.Mol)
        if not isinstance(mols[0], Chem.Mol):
            # necessary for mypy
            raise AssertionError("Expected a Chem.Mol object")
        self.assertEqual(mols[0].GetNumAtoms(), 8)

        # test multiple sdf strings with multiple molecules
        # the current behavior is to return the first molecule only
        mols = SDFToMol().transform(
            [self.sdf_str_benzaldehyde_aspirin, self.sdf_str_benzaldehyde],
        )
        self.assertEqual(len(mols), 2)
        self.assertIsInstance(mols[0], Chem.Mol)
        self.assertIsInstance(mols[1], Chem.Mol)
        if not isinstance(mols[0], Chem.Mol):
            # necessary for mypy
            raise AssertionError("Expected a Chem.Mol object")
        self.assertEqual(mols[0].GetNumAtoms(), 8)
        if not isinstance(mols[1], Chem.Mol):
            # necessary for mypy
            raise AssertionError("Expected a Chem.Mol object")
        self.assertEqual(mols[1].GetNumAtoms(), 8)

    def test_sdf_properties_transfer(self) -> None:
        """Test that properties from SDF are transferred to molecule.

        Raises
        ------
        AssertionError
            If the transformation does not return a valid molecule.

        """
        mol = SDFToMol().pretransform_single(self.sdf_str_benzaldehyde)
        if not isinstance(mol, Chem.Mol):
            # necessary for mypy
            raise AssertionError("Expected a Chem.Mol object")
        self.assertTrue(mol.HasProp("PUBCHEM_COMPOUND_CID"))
        self.assertEqual(mol.GetProp("PUBCHEM_COMPOUND_CID"), "240")
        self.assertTrue(mol.HasProp("PUBCHEM_COMPOUND_CANONICALIZED"))
        self.assertEqual(mol.GetProp("PUBCHEM_COMPOUND_CANONICALIZED"), "1")
        self.assertTrue(mol.HasProp("PUBCHEM_CACTVS_COMPLEXITY"))
        self.assertEqual(mol.GetProp("PUBCHEM_CACTVS_COMPLEXITY"), "72.5")
