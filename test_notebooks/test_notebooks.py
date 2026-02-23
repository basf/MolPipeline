"""Script that runs all notebooks once to check if they execute without error."""

import unittest
from pathlib import Path

import nbformat
from loguru import logger
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

# directory containing notebooks
NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks"

# We mainly skip because some notebooks run too long.
SKIP_NOTEBOOK_LIST = ["notebooks/advanced_01_hyperopt_on_bbbp.ipynb"]


def get_notebook_paths_from_dir(notebook_dir: Path) -> list[Path]:
    """Get all Jupyter notebook files in the directory.

    Parameters
    ----------
    notebook_dir: Path
        Path to the directory containing the Jupyter notebooks.

    Returns
    -------
    list[Path]
        List of paths to Jupyter notebooks.

    """
    # Find all Jupyter notebook files in the directory
    notebooks_paths = []
    for notebook_path in notebook_dir.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in str(notebook_path.resolve()):
            # skip jetbrains checkpoints
            continue
        if notebook_path.name.endswith(".nbconvert.ipynb"):
            # skip converted notebooks
            continue

        notebooks_paths.append(notebook_path)
    return notebooks_paths


class TestNotebooks(unittest.TestCase):
    """Test class for testing if all Jupyter notebooks run through with error code 0."""

    def run_notebook(
        self,
        notebook_path: Path,
    ) -> None:
        """Check all Jupyter notebooks in the given directory.

        Parameters
        ----------
        notebook_path: Path
            Path to the Jupyter notebook to run.

        """
        logger.info(f"Running notebook: {notebook_path}")

        with notebook_path.open(encoding="UTF-8") as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

        try:
            ep.preprocess(nb, {"metadata": {"path": notebook_path.parent}})
        except CellExecutionError as e:
            logger.error(f"Error executing {notebook_path}.")
            logger.error(str(e))
            self.fail(f"Error executing {notebook_path}. Error: {e!s}")

    def test_notebooks(self) -> None:
        """Test if all Jupyter notebooks run through with error code 0."""
        notebooks_paths_list = get_notebook_paths_from_dir(NOTEBOOK_DIR)
        for notebooks_path in notebooks_paths_list:
            if str(notebooks_path) in SKIP_NOTEBOOK_LIST:
                logger.warning(f"Skipping notebook: {notebooks_path}")
                continue
            with self.subTest(notebook=notebooks_path):
                self.run_notebook(notebooks_path)


if __name__ == "__main__":
    unittest.main()
