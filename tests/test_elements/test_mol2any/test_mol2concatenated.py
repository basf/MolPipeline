"""Test construction of concatenated fingerprints."""

from __future__ import annotations

import itertools
import unittest
from typing import Any, Literal, get_args

import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import (
    MolToConcatenatedVector,
    MolToMorganFP,
    MolToNetCharge,
    MolToRDKitPhysChem,
)
from tests.utils.fingerprints import fingerprints_to_numpy


class TestConcatenatedFingerprint(unittest.TestCase):
    """Unittest for MolToConcatenatedVector, which calculates concatenated fingerprints."""

    def test_generation(self) -> None:
        """Test if the feature concatenation works as expected."""
        fingerprint_morgan_output_types: tuple[Any, ...] = get_args(
            Literal[
                "sparse",
                "dense",
                "explicit_bit_vect",
            ]
        )

        smiles = [
            "CC",
            "CCC",
            "CCCO",
            "CCNCO",
            "C(C)CCO",
            "CCO",
            "CCCN",
            "CCCC",
            "CCOC",
            "COO",
        ]

        for fp_output_type in fingerprint_morgan_output_types:

            concat_vector_element = MolToConcatenatedVector(
                [
                    (
                        "RDKitPhysChem",
                        MolToRDKitPhysChem(standardizer=StandardScaler()),
                    ),
                    (
                        "MorganFP",
                        MolToMorganFP(return_as=fp_output_type),
                    ),
                ]
            )
            pipeline = Pipeline(
                [
                    ("smi2mol", SmilesToMol()),
                    ("concat_vector_element", concat_vector_element),
                ]
            )

            output = pipeline.fit_transform(smiles)
            output2 = pipeline.transform(smiles)

            mol_list = [Chem.MolFromSmiles(smi) for smi in smiles]
            output3 = np.hstack(
                [
                    concat_vector_element.element_list[0][1].transform(mol_list),
                    fingerprints_to_numpy(
                        concat_vector_element.element_list[1][1].transform(mol_list)
                    ),
                ]
            )
            pyschem_component: MolToRDKitPhysChem
            pyschem_component = concat_vector_element.element_list[0][1]  # type: ignore
            morgan_component: MolToMorganFP
            morgan_component = concat_vector_element.element_list[1][1]  # type: ignore
            expected_shape = (
                len(smiles),
                (pyschem_component.n_features + morgan_component.n_bits),
            )
            self.assertEqual(output.shape, expected_shape)
            self.assertTrue(np.allclose(output, output2))
            self.assertTrue(np.allclose(output, output3))

    def test_empty_element_list(self) -> None:
        """Test if an empty element list raises an error."""
        with self.assertRaises(ValueError):
            MolToConcatenatedVector([])

    def test_n_features(self) -> None:
        """Test getting the number of features in the concatenated vector."""

        physchem_elem = (
            "RDKitPhysChem",
            MolToRDKitPhysChem(),
        )
        morgan_elem = (
            "MorganFP",
            MolToMorganFP(n_bits=16),
        )
        net_charge_elem = ("NetCharge", MolToNetCharge())

        self.assertEqual(
            MolToConcatenatedVector([physchem_elem]).n_features,
            physchem_elem[1].n_features,
        )
        self.assertEqual(
            MolToConcatenatedVector([morgan_elem]).n_features,
            16,
        )
        self.assertEqual(
            MolToConcatenatedVector([net_charge_elem]).n_features,
            net_charge_elem[1].n_features,
        )
        self.assertEqual(
            MolToConcatenatedVector([physchem_elem, morgan_elem]).n_features,
            physchem_elem[1].n_features + 16,
        )
        self.assertEqual(
            MolToConcatenatedVector([net_charge_elem, morgan_elem]).n_features,
            net_charge_elem[1].n_features + 16,
        )
        self.assertEqual(
            MolToConcatenatedVector(
                [net_charge_elem, morgan_elem, physchem_elem]
            ).n_features,
            net_charge_elem[1].n_features + 16 + physchem_elem[1].n_features,
        )

    def test_features_names(self) -> None:  # pylint: disable=too-many-arguments
        """Test getting the names of features in the concatenated vector."""

        physchem_elem = (
            "RDKitPhysChem",
            MolToRDKitPhysChem(),
        )
        net_charge_elem = ("NetCharge", MolToNetCharge())
        morgan_elem = (
            "MorganFP",
            MolToMorganFP(n_bits=16),
        )
        path_elem = (
            "PathFP",
            MolToMorganFP(n_bits=15),
        )
        maccs_elem = (
            "MACCSFP",
            MolToMorganFP(n_bits=14),
        )

        elements = [physchem_elem, net_charge_elem, morgan_elem, path_elem, maccs_elem]

        for feature_names_prefix in [None, "my_prefix"]:
            # test all subsets are compatible
            powerset = itertools.chain.from_iterable(
                itertools.combinations(elements, r) for r in range(len(elements) + 1)
            )
            # skip empty subset
            next(powerset)

            for elements_subset in powerset:
                conc_elem = MolToConcatenatedVector(
                    list(elements_subset), feature_names_prefix=feature_names_prefix
                )
                feature_names = conc_elem.feature_names

                # test a feature names and n_features are consistent
                self.assertEqual(
                    len(feature_names),
                    conc_elem.n_features,
                )

                seen_names = 0
                for elem_name, elem in elements_subset:
                    self.assertTrue(hasattr(elem, "feature_names"))
                    elem_feature_names = elem.feature_names  # type: ignore[attr-defined]
                    elem_n_features = len(elem_feature_names)
                    relevant_names = feature_names[
                        seen_names : seen_names + elem_n_features
                    ]
                    prefixes, feat_names = map(
                        list, zip(*[name.split("__") for name in relevant_names])
                    )
                    # test feature names are the same
                    self.assertListEqual(elem_feature_names, feat_names)

                    if feature_names_prefix is not None:
                        # test prefixes are the same user given prefix
                        self.assertTrue(
                            all(prefix == feature_names_prefix for prefix in prefixes)
                        )
                    else:
                        # test prefixes are the same as element names
                        self.assertTrue(all(prefix == elem_name for prefix in prefixes))

                    seen_names += elem_n_features


if __name__ == "__main__":
    unittest.main()
