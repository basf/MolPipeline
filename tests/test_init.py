"""Test functionality set at package init."""

import unittest

from molpipeline import __version__


class TestInit(unittest.TestCase):
    """Test functionality set at package init."""

    def test_version(self) -> None:
        """Test that the package has a version."""
        self.assertIsInstance(__version__, str)
        splitted = __version__.split(".")
        self.assertEqual(len(splitted), 3)
        major, minor, patch = splitted
        self.assertTrue(major.isdigit())
        self.assertTrue(minor.isdigit())
        self.assertTrue(patch.isdigit())
