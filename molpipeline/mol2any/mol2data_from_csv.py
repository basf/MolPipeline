"""Element that reads features/descriptors from a file."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from molpipeline.abstract_pipeline_elements.core import (
    MolToAnyPipelineElement,
    InvalidInstance,
)

from molpipeline.mol2any import MolToSmiles, MolToInchi, MolToInchiKey
from molpipeline.utils.molpipeline_types import RDKitMol


def _mol_to_identifier(mol: RDKitMol, id_type: str) -> str:
    """Convert a molecule to its identifier.

    Parameters
    ----------
    mol: RDKitMol
        Molecule to convert
    id_type: str
        Type of identifier to use. Can be "smiles", "inchi", or "inchikey".

    Returns
    -------
    str
        Identifier for the molecule

    Raises
    ------
    ValueError
        If id_type is not one of "smiles", "inchi", or "inchikey"
    """
    if id_type == "smiles":
        return MolToSmiles().transform_single(mol)
    elif id_type == "inchi":
        return MolToInchi().transform_single(mol)
    elif id_type == "inchikey":
        return MolToInchiKey().transform_single(mol)
    else:
        raise ValueError(f"Invalid id_type: {id_type}")


class MolToDataFromCSV(MolToAnyPipelineElement):
    """Pipeline element that reads precalculated descriptors from a CSV file.

    Maps molecules to their descriptors using an identifier column (e.g. SMILES, InChI).
    """

    def __init__(
        self,
        feature_file_path: str | Path,
        identifier_column: str,
        feature_columns: list[str],
        id_type: Literal["smiles", "inchi", "inchikey"] = "smiles",
        missing_value_strategy: Literal["invalid_instance", "nan"] = "invalid_instance",
        name: str = "MolToFeaturesFromFile",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize MolToFeaturesFromFile.

        Parameters
        ----------
        feature_file_path: str | Path
            Path to the file containing precalculated features
        identifier_column: str
            Name of the column containing molecule identifiers
        feature_columns: list[str]
            List of column names to extract as features
        id_type: Literal["smiles", "inchi", "inchikey"], optional
            Type of identifier to use for molecule matching. Default is "smiles"
        missing_value_strategy: Literal["invalid_instance", "nan"], optional
            Strategy for handling missing values. Default is "invalid_instance"
        name: str, optional
            Name of the pipeline element. Default is "MolToFeaturesFromFile"
        n_jobs: int, optional
            Number of parallel jobs. Default is 1
        uuid: str | None, optional
            UUID of the pipeline element

        Raises
        ------
        ValueError
            If feature_columns is empty
        FileNotFoundError
            If feature_file_path doesn't exist
        """
        if not feature_columns:
            raise ValueError("Empty feature_columns is not allowed")

        self.feature_file_path = Path(feature_file_path)
        self.identifier_column = identifier_column
        self.feature_columns = feature_columns
        self.id_type = id_type
        self.missing_value_strategy = missing_value_strategy

        if not self.feature_file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {self.feature_file_path}")

        self.features_df = MolToDataFromCSV._read_data_table(
            self.feature_file_path,
            self.identifier_column,
            self.feature_columns,
        )

        # TODO check for uniqueness of identifier_column. Drop duplicates if necessary

        # Validate columns existence
        missing_cols = set(self.feature_columns) - set(self.features_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in feature file: {missing_cols}")

        # # Create lookup dictionary for faster access
        # self.id_to_features = {
        #     id_val: self.features_df.loc[idx, self.feature_columns].values
        #     for idx, id_val in enumerate(self.features_df[self.identifier_column])
        # }

        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)

    @staticmethod
    def _read_data_table(
        feature_file_path: Path,
        identifier_column: str,
        feature_columns: Sequence[str],
    ) -> pd.DataFrame:
        sep = ","
        if feature_file_path.name.endswith(".tsv"):
            sep = "\t"

        try:
            dtype_dict: dict[str, Any] = {col: np.float64 for col in feature_columns}
            dtype_dict[identifier_column] = str
            usecols = list(dtype_dict.keys())
            return pd.read_csv(
                feature_file_path,
                index_col=identifier_column,
                usecols=usecols,
                dtype=dtype_dict,
                sep=sep,
            )
        except Exception as e:
            raise ValueError(f"Error reading feature file: {e}") from e

    def pretransform_single(
        self, value: RDKitMol
    ) -> npt.NDArray[np.float64] | InvalidInstance:
        """Transform a molecule to its features from the file.

        Parameters
        ----------
        value: RDKitMol
            Molecule to transform

        Returns
        -------
        npt.NDArray[np.float64] | InvalidInstance
            Features for the molecule or InvalidInstance if not found and missing_value_strategy is "invalid_instance"
        """
        try:
            # Convert molecule to identifier
            mol_id = _mol_to_identifier(value, self.id_type)

            # Look up features
            if mol_id in self.features_df.index:
                # Get features as numpy array
                return self.features_df.loc[mol_id, self.feature_columns].to_numpy(
                    dtype=np.float64
                )

            # Handle missing values
            if self.missing_value_strategy == "invalid_instance":
                return InvalidInstance(
                    self.uuid,
                    f"No features found for molecule with {self.id_type}: {mol_id}",
                )
            else:  # "nan"
                return np.full(len(self.feature_columns), np.nan)

        except Exception as e:
            warnings.warn(f"Error processing molecule: {e}", UserWarning, stacklevel=2)
            return InvalidInstance(self.uuid, f"Error processing molecule: {str(e)}")

    def assemble_output(
        self, value_list: Iterable[npt.NDArray[np.float64] | InvalidInstance]
    ) -> list[npt.NDArray[np.float64] | InvalidInstance]:
        """Assemble output from pretransform_single.

        Parameters
        ----------
        value_list: Iterable
            List of transformed values

        Returns
        -------
        list
            List of features arrays or InvalidInstance objects
        """
        return np.array(list(value_list))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this pipeline element.

        Parameters
        ----------
        deep: bool, default=True
            If True, return parameters of subobjects

        Returns
        -------
        dict
            Parameters of this pipeline element
        """
        params = super().get_params(deep=deep)
        params.update(
            {
                "feature_file_path": self.feature_file_path,
                "identifier_column": self.identifier_column,
                "feature_columns": self.feature_columns,
                "id_type": self.id_type,
                "missing_value_strategy": self.missing_value_strategy,
            }
        )
        return params

    def set_params(self, **parameters: Any) -> MolToDataFromCSV:
        """Set parameters of this pipeline element.

        Parameters
        ----------
        **parameters
            Parameters to set

        Returns
        -------
        MolToDataFromCSV
            Pipeline element with parameters set
        """
        super().set_params(**parameters)
        return self
