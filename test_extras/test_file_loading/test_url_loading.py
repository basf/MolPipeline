"""Tests for URL file loading utilities."""

import unittest
from unittest import mock

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
        new_timeout = 20
        self.loader.set_params(url=new_url, timeout=new_timeout)
        self.assertEqual(self.loader.url, new_url)
        self.assertEqual(self.loader.timeout, new_timeout)

    def test_load_file(self) -> None:
        """Test load_file method."""
        mock_message = "Mocked content"
        with mock.patch("requests.get") as mock_get:
            mock_response = mock.MagicMock()
            mock_response.content = mock_message.encode("utf-8")
            mock_response.raise_for_status = mock.MagicMock()
            mock_get.return_value = mock_response
            loaded_file = self.loader.load_file()
            self.assertEqual(mock_get.call_count, 1)

        self.assertEqual(loaded_file.decode(), mock_message)
