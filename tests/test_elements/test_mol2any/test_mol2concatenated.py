"""Test construction of concatenated fingerprints."""

from __future__ import annotations

import itertools
import unittest
from typing import Any, Literal, get_args

import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

from molpipeline import Pipeline
from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import (
    Mol2PathFP,
    MolToConcatenatedVector,
    MolToMACCSFP,
    MolToMorganFP,
    MolToNetCharge,
    MolToRDKitPhysChem,
)
from tests.utils.fingerprints import fingerprints_to_numpy
from tests.utils.logging import capture_logs


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
        # test constructor
        with self.assertRaises(ValueError):
            MolToConcatenatedVector([])

        # test setter
        concat_elem = MolToConcatenatedVector(
            [
                (
                    "RDKitPhysChem",
                    MolToRDKitPhysChem(),
                )
            ]
        )
        with self.assertRaises(ValueError):
            concat_elem.set_params(element_list=[])

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

    def test_features_names(self) -> None:  # pylint: disable-msg=too-many-locals
        """Test getting the names of features in the concatenated vector."""

        physchem_elem = (
            "RDKitPhysChem",
            MolToRDKitPhysChem(),
        )
        net_charge_elem = ("NetCharge", MolToNetCharge())
        morgan_elem = (
            "MorganFP",
            MolToMorganFP(n_bits=17),
        )
        path_elem = (
            "PathFP",
            Mol2PathFP(n_bits=15),
        )
        maccs_elem = (
            "MACCSFP",
            MolToMACCSFP(),
        )

        elements = [
            physchem_elem,
            net_charge_elem,
            morgan_elem,
            path_elem,
            maccs_elem,
        ]

        for use_feature_names_prefix in [False, True]:
            # test all subsets are compatible
            powerset = itertools.chain.from_iterable(
                itertools.combinations(elements, r) for r in range(len(elements) + 1)
            )
            # skip empty subset
            next(powerset)

            for elements_subset in powerset:
                conc_elem = MolToConcatenatedVector(
                    list(elements_subset),
                    use_feature_names_prefix=use_feature_names_prefix,
                )
                feature_names = conc_elem.feature_names

                if use_feature_names_prefix:
                    # test feature names are unique if prefix is used or only one element is used
                    self.assertEqual(
                        len(feature_names),
                        len(set(feature_names)),
                    )

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

                    if use_feature_names_prefix:
                        # feature_names should be prefixed with element name
                        prefixes, feat_names = map(
                            list, zip(*[name.split("__") for name in relevant_names])
                        )
                        # test feature names are the same
                        self.assertListEqual(elem_feature_names, feat_names)
                        # test prefixes are the same as element names
                        self.assertTrue(all(prefix == elem_name for prefix in prefixes))
                    else:
                        # feature_names should be the same as element feature names
                        self.assertListEqual(elem_feature_names, relevant_names)

                    seen_names += elem_n_features

    def test_logging_feature_names_uniqueness(self) -> None:
        """Test that a warning is logged when feature names are not unique."""
        elements: list[tuple[str, MolToAnyPipelineElement]] = [
            (
                "MorganFP",
                MolToMorganFP(n_bits=17),
            ),
            (
                "MorganFP_with_feats",
                MolToMorganFP(n_bits=16, use_features=True),
            ),
        ]

        # First test is with no prefix
        use_feature_names_prefix = False
        with capture_logs() as output:
            conc_elem = MolToConcatenatedVector(
                elements,
                use_feature_names_prefix=use_feature_names_prefix,
            )
            feature_names = conc_elem.feature_names

        # test log message
        self.assertEqual(len(output), 1)
        message = output[0]
        self.assertIn(
            "Feature names in MolToConcatenatedVector are not unique", message
        )
        self.assertEqual(message.record["level"].name, "WARNING")

        # test feature names are NOT unique
        self.assertNotEqual(len(feature_names), len(set(feature_names)))

        # Second test is with prefix
        use_feature_names_prefix = True
        with capture_logs() as output:
            conc_elem = MolToConcatenatedVector(
                elements,
                use_feature_names_prefix=use_feature_names_prefix,
            )
            feature_names = conc_elem.feature_names

        # test log message
        self.assertEqual(len(output), 0)

        # test feature names are unique
        self.assertEqual(len(feature_names), len(set(feature_names)))

    def test_getter_setter(self) -> None:
        """Test getter and setter methods."""
        elements: list[tuple[str, MolToAnyPipelineElement]] = [
            (
                "MorganFP",
                MolToMorganFP(n_bits=17),
            ),
            (
                "MorganFP_with_feats",
                MolToMorganFP(n_bits=16, use_features=True),
            ),
        ]
        concat_elem = MolToConcatenatedVector(
            elements,
            use_feature_names_prefix=True,
        )
        self.assertEqual(len(concat_elem.get_params()["element_list"]), 2)
        self.assertEqual(concat_elem.get_params()["use_feature_names_prefix"], True)
        # test that there are no duplicates in feature names
        self.assertEqual(
            len(concat_elem.feature_names), len(set(concat_elem.feature_names))
        )
        params: dict[str, Any] = {
            "use_feature_names_prefix": False,
        }
        concat_elem.set_params(**params)
        self.assertEqual(concat_elem.get_params()["use_feature_names_prefix"], False)
        # test that there are duplicates in feature names
        self.assertNotEqual(
            len(concat_elem.feature_names), len(set(concat_elem.feature_names))
        )


if __name__ == "__main__":
    unittest.main()
