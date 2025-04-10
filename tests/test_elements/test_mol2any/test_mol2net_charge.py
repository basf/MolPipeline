"""Test generation of net charge calculation."""

import unittest

import numpy as np
import pandas as pd

from molpipeline import ErrorFilter, FilterReinserter, Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToNetCharge

DF_TEST_DATA = pd.DataFrame(
    {
        "smiles": [
            "[Fe+2]",
            "c1cc(c(nc1)Cl)C(=O)Nc2c(c3c(s2)CCCCC3)C(=O)N",
            "Cc1ccc(cc1)S(=O)(=O)Nc2c(c3c(s2)C[C@@H](CC3)C)C(=O)N",
            "c1cc(oc1)CN=C2C=C(C(=CC2=C(O)[O-])S(=O)(=O)[NH-])Cl",
            "C[C@@H]1[C@@H](OP2(O1)(O[C@H]([C@H](O2)C)C)C[NH+]3CCCCC3)C",  # this one fails gasteiger charge computation
        ],
        "expected_net_charges_formal_charge": [2, 0, 0, -2, 1],
        "expected_net_charges_gasteiger": [2, -1, -1, -2, np.nan],
    }
)


class TestNetChargeCalculator(unittest.TestCase):
    """Unittest for MolToNetCharge, which calculates net charges of molecules."""

    def test_net_charge_calculation_formal_charge(self) -> None:
        """Test if the net charge calculation works as expected for formal charges.

        Returns
        -------
        None
        """
        # we need the error filter and reinserter to handle the case where the charge calculation fails
        error_filter = ErrorFilter(filter_everything=True)
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "net_charge_element",
                    MolToNetCharge(charge_method="formal_charge", standardizer=None),
                ),
                ("error_filter", error_filter),
                (
                    "filter_reinserter",
                    FilterReinserter.from_error_filter(error_filter, fill_value=np.nan),
                ),
            ],
        )

        actual_net_charges = pipeline.fit_transform(DF_TEST_DATA["smiles"])
        self.assertTrue(
            np.allclose(
                DF_TEST_DATA["expected_net_charges_formal_charge"]
                .to_numpy()
                .reshape(-1, 1),
                actual_net_charges,
                equal_nan=True,
            )
        )

    def test_net_charge_calculation_gasteiger(self) -> None:
        """Test if the net charge calculation works as expected for gasteiger charges.

        Returns
        -------
        None
        """
        # we need the error filter and reinserter to handle the case where the charge calculation fails
        error_filter = ErrorFilter(filter_everything=True)
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "net_charge_element",
                    MolToNetCharge(charge_method="gasteiger", standardizer=None),
                ),
                ("error_filter", error_filter),
                (
                    "filter_reinserter",
                    FilterReinserter.from_error_filter(error_filter, fill_value=np.nan),
                ),
            ],
        )

        actual_net_charges = pipeline.fit_transform(DF_TEST_DATA["smiles"])
        self.assertTrue(
            np.allclose(
                DF_TEST_DATA["expected_net_charges_gasteiger"]
                .to_numpy()
                .reshape(-1, 1),
                actual_net_charges,
                equal_nan=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
