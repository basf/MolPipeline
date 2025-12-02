"""Tests for URL file loading utilities."""

import unittest

from molpipeline.utils.file_loading.url_file_loading import URLFileLoader


class TestURLFileLoader(unittest.TestCase):
    """Unit tests for the URLFileLoader class."""

    def setUp(self) -> None:
        """Set up test variables."""
        self.test_url = (
            "https://github.com/basf/MolPipeline/blob/main/.github/molpipeline.png"
        )
        self.loader = URLFileLoader(url=self.test_url, timeout=10)

    def test_initialization(self) -> None:
        """Test initialization of URLFileLoader."""
        self.assertEqual(self.loader.url, self.test_url)
        self.assertEqual(self.loader.timeout, 10)

    def test_get_params(self) -> None:
        """Test get_params method."""
        params = self.loader.get_params()
        self.assertEqual(params, {"url": self.test_url, "timeout": 10})

    def test_set_params(self) -> None:
        """Test set_params method."""
        new_url = "https://www.test.com"
        self.loader.set_params(url=new_url)
        self.assertEqual(self.loader.url, new_url)

    def test_load_file(self) -> None:
        """Test load_file method."""
        content = self.loader.load_file()
        self.assertIsInstance(content, bytes)
        self.assertGreater(len(content), 0)
