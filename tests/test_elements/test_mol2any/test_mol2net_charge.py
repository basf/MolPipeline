"""Test generation of net charge calculation."""

import unittest

import numpy as np
import pandas as pd

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToNetCharge


class TestNetChargeCalculator(unittest.TestCase):
    """Unittest for MolToNetCharge, which calculates net charges of molecules."""

    def test_net_charge_calculation(self) -> None:
        """Test if the net charge calculation works as expected.

        Returns
        -------
        None
        """
        df = pd.DataFrame(
            {
                "smiles": [
                    "[Fe+2]",
                    "c1cc(c(nc1)Cl)C(=O)Nc2c(c3c(s2)CCCCC3)C(=O)N",
                    "Cc1ccc(cc1)S(=O)(=O)Nc2c(c3c(s2)C[C@@H](CC3)C)C(=O)N",
                    "c1cc(oc1)CN=C2C=C(C(=CC2=C(O)[O-])S(=O)(=O)[NH-])Cl",
                    "C[C@@H]1[C@@H](OP2(O1)(O[C@H]([C@H](O2)C)C)C[NH+]3CCCCC3)C",  # this one fails charge computation
                ],
                "expected_net_charges": [2, -1, -1, -2, np.nan],
            }
        )

        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("net_charge_element", MolToNetCharge(standardizer=None)),
            ],
        )

        actual_net_charges = pipeline.fit_transform(df["smiles"])
        self.assertTrue(
            np.allclose(
                df["expected_net_charges"].to_numpy().reshape(-1, 1), actual_net_charges
            )
        )


if __name__ == "__main__":
    unittest.main()
