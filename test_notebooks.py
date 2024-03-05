"""Script that runs all notebooks once to check if they execute without error."""

import argparse
import subprocess
import sys
from pathlib import Path

# list of directories containing notebooks
NOTEBOOK_DIRS = ["notebooks"]

# list of prefix paths of notebooks to skip. We mainly skip because some notebooks run too long.
SKIP_NOTEBOOKS_PREFIXES = ["notebooks/advanced_01"]


def is_prefix(prefix: Path, path: Path) -> bool:
    """Check if the prefix is a prefix of the path.

    Parameters
    ----------
    prefix: Path
        Prefix path.
    path: Path
        Path to check if it has the prefix.

    Returns
    -------
    bool
        True if the prefix is a prefix of the path, False otherwise.
    """
    # compares the parents of the paths are equal and the name of the path starts with the prefix file name
    return all(
        parent1 == parent2
        for parent1, parent2 in zip(
            prefix.resolve().parents, path.resolve().parents, strict=True
        )
    ) and path.name.startswith(prefix.name)


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


def filter_notebooks(
    notebooks_paths: list[Path], skip_notebooks_prefixes: list[Path]
) -> list[Path]:
    """Filter out notebooks that should be skipped.

    Parameters
    ----------
    notebooks_paths: list[Path]
        List of paths to Jupyter notebooks.
    skip_notebooks_prefixes: list[Path]
        List of prefixes of notebooks to skip.

    Returns
    -------
    list[Path]
        Filtered list of paths to Jupyter notebooks.
    """
    return list(
        filter(
            lambda notebook_path: all(
                not is_prefix(prefix, notebook_path)
                for prefix in skip_notebooks_prefixes
            ),
            notebooks_paths,
        )
    )


def run_notebooks(
    notebook_dir: Path, skip_notebooks_prefixes: list[Path], continue_on_error: bool
) -> None:
    """Run all Jupyter notebooks in the given directory and check if they execute without error.

    Parameters
    ----------
    notebook_dir: Path
        Path to the directory containing the Jupyter notebooks.
    skip_notebooks_prefixes: list[Path]
        List of prefixes of notebooks to skip.
    continue_on_error: bool
        If True continue running all notebooks even if an error occurs.
    """
    # collect all notebook files from disc
    notebooks_paths = get_notebook_paths_from_dir(notebook_dir)

    # filter out notebooks that should be skipped
    notebooks_paths = sorted(filter_notebooks(notebooks_paths, skip_notebooks_prefixes))

    nof_errors = 0
    # Loop through each notebook
    for notebooks_path in notebooks_paths:

        # Execute the notebook and capture the error code
        cmd = [
            "jupyter",
            "nbconvert",
            "--execute",
            str(notebooks_path),
            "--to",
            "notebook",
        ]
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            std_out, std_err = process.communicate()
            error_code = process.returncode

        # Check if the error code is not 0
        if error_code != 0:
            nof_errors += 1
            log_error_msg = (
                f"Error executing {notebooks_path}. Error code: {error_code}"
            )
            print(log_error_msg)
            print(std_out.decode("utf-8"))
            print(std_err.decode("utf-8"))

            if not continue_on_error:
                sys.exit(1)

    # Check if there were any errors
    if nof_errors == 0:
        print("All Jupyter notebooks executed successfully with error code 0.")
    else:
        print(f"Jupyter notebooks executed with {nof_errors} errors.")
        sys.exit(1)


def main() -> None:
    """Main function to run the Jupyter notebooks."""
    parser = argparse.ArgumentParser(
        description="Test if all Jupyter notebooks in a directory run through with error code 0"
    )
    parser.add_argument(
        "-c",
        "--continue-on-failure",
        dest="continue_on_error",
        action="store_true",
        help="Continue running all notebooks even if an error occurs",
    )
    parser.set_defaults(continue_on_error=False)
    args = parser.parse_args()

    skip_notebook_prefixes_paths = [Path(prefix) for prefix in SKIP_NOTEBOOKS_PREFIXES]

    for notebook_dir in NOTEBOOK_DIRS:
        run_notebooks(
            Path(notebook_dir), skip_notebook_prefixes_paths, args.continue_on_error
        )


if __name__ == "__main__":
    main()
