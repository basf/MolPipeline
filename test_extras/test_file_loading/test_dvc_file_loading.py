"""Test DVC file loading functionality."""

import io
import unittest

import pandas as pd

from molpipeline.utils.file_loading.dvc_file_loading import DVCFileLoader


class TestDVCFileLoading(unittest.TestCase):
    """Test DVC file loading functionality."""

    def setUp(self) -> None:
        """Set up test variables."""
        self.file_path = "tests/test_data/mol_descriptors.tsv"
        self.repo = "https://github.com/basf/MolPipeline"
        self.rev = "main"
        self.dvc_loader = DVCFileLoader(
            file_path=self.file_path,
            repo=self.repo,
            rev=self.rev,
        )

    def test_load_file(self) -> None:
        """Test loading a file via DVC."""
        content = self.dvc_loader.load_file()
        self.assertIsInstance(content, bytes)
        self.assertGreater(len(content), 0)
        loaded_df = pd.read_csv(io.BytesIO(content), sep="\t")
        self.assertFalse(loaded_df.empty)

    def test_get_params(self) -> None:
        """Test getting parameters of the DVC file loader."""
        params = self.dvc_loader.get_params()
        self.assertEqual(params["file_path"], self.file_path)
        self.assertEqual(params["repo"], self.repo)
        self.assertEqual(params["rev"], self.rev)

    def test_set_params(self) -> None:
        """Test setting parameters of the DVC file loader."""
        new_file_path = "Readme.md"
        new_repo = "https://github.com/EBjerrum/scikit-mol"
        new_rev = "dev"
        dvc_loader = DVCFileLoader(
            file_path=self.file_path,
            repo=self.repo,
            rev=self.rev,
        )
        dvc_loader.set_params(
            file_path=new_file_path,
            repo=new_repo,
            rev=new_rev,
        )
        params = dvc_loader.get_params()
        self.assertEqual(params["file_path"], new_file_path)
        self.assertEqual(params["repo"], new_repo)
        self.assertEqual(params["rev"], new_rev)
