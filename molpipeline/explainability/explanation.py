"""Module for explanation class."""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

from molpipeline.abstract_pipeline_elements.core import RDKitMol


@dataclasses.dataclass()
class Explanation:
    """Class representing explanations of a prediction."""

    # input data
    feature_vector: npt.NDArray[np.float_] | None = None
    feature_names: list[str] | None = None
    molecule: RDKitMol | None = None
    prediction: float | npt.NDArray[np.float_] | None = None

    # explanation results mappable to the feature vector
    feature_weights: npt.NDArray[np.float_] | None = None

    # explanation results mappable to the molecule.
    atom_weights: npt.NDArray[np.float_] | None = None
    bond_weights: npt.NDArray[np.float_] | None = None

    def is_valid(self) -> bool:
        """Check if the explanation is valid.

        Returns
        -------
        bool
            True if the explanation is valid, False otherwise.
        """
        return all(
            [
                self.feature_vector is not None,
                # self.feature_names is not None,
                self.molecule is not None,
                self.prediction is not None,
                any(
                    [
                        self.feature_weights is not None,
                        self.atom_weights is not None,
                        self.bond_weights is not None,
                    ]
                ),
            ]
        )
