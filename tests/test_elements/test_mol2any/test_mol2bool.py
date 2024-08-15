"""Test mol to bool conversion."""

import unittest

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.mol2any.mol2bool import MolToBool


class TestMolToBool(unittest.TestCase):
    """Unittest for MolToBool."""

    def test_bool_conversion(self) -> None:
        """Test if the invalid instances are converted to bool."""

        mol2bool = MolToBool()
        result = mol2bool.transform(
            [
                1,
                2,
                InvalidInstance(element_id="test", message="test", element_name="Test"),
                4,
            ]
        )
        self.assertEqual(result, [True, True, False, True])