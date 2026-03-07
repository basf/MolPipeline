"""Unit tests for CloneEnsembleClassifier and CloneEnsembleRegressor."""

import unittest

from molpipeline.estimators.ensemble.model_clone_ensemble import (
    CloneEnsembleRegressor,
)
from tests.templates.test_wrapped_estimators import WrappedEstimatorBaseTestMixIn


class TestCloneEnsembleRegressor(unittest.TestCase, WrappedEstimatorBaseTestMixIn):
    """Unit tests for CloneEnsembleRegressor."""

    @staticmethod
    def get_wrapped_estimator_type() -> type:
        """Return the CloneEnsembleRegressor class.

        Returns
        -------
        type[CloneEnsembleRegressor]
                The class of the wrapped estimator to be tested.

        """
        return CloneEnsembleRegressor

    @staticmethod
    def get_test_parameters() -> dict[str, int]:
        """Return a dictionary of parameters to be used for testing.

        Returns
        -------
        dict[str, int]
            A dictionary of parameters to be used for testing.

        """
        return {"n_estimators": 3}


class TestCloneEnsembleClassifier(unittest.TestCase, WrappedEstimatorBaseTestMixIn):
    """Unit tests for CloneEnsembleClassifier."""

    @staticmethod
    def get_wrapped_estimator_type() -> type:
        """Return the CloneEnsembleRegressor class.

        Returns
        -------
        type[CloneEnsembleRegressor]
            The class of the wrapped estimator to be tested.

        """
        return CloneEnsembleRegressor

    @staticmethod
    def get_test_parameters() -> dict[str, int]:
        """Return a dictionary of parameters to be used for testing.

        Returns
        -------
        dict[str, int]
            A dictionary of parameters to be used for testing.

        """
        return {"n_estimators": 3}


if __name__ == "__main__":
    unittest.main()
