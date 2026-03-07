"""Unit tests for SplitEnsembleClassifier and SplitEnsembleRegressor."""

import unittest

from molpipeline.estimators.ensemble.split_ensemble import (
    SplitEnsembleClassifier,
    SplitEnsembleRegressor,
)
from tests.templates.test_wrapped_estimators import WrappedEstimatorBaseTestMixIn


class TestSplitEnsembleRegressor(unittest.TestCase, WrappedEstimatorBaseTestMixIn):
    """Unit tests for SplitEnsembleRegressor."""

    @staticmethod
    def get_wrapped_estimator_type() -> type:
        """Return the SplitEnsembleRegressor class.

        Returns
        -------
        type[SplitEnsembleRegressor]
            The SplitEnsembleRegressor class.

        """
        return SplitEnsembleRegressor

    @staticmethod
    def get_test_parameters() -> dict[str, int]:
        """Return a dictionary of parameters to be used for testing.

        Returns
        -------
        dict[str, int]
            A dictionary of parameters to be used for testing.

        """
        return {"cv": 2}


class TestSplitEnsembleClassifier(unittest.TestCase, WrappedEstimatorBaseTestMixIn):
    """Unit tests for SplitEnsembleClassifier."""

    @staticmethod
    def get_wrapped_estimator_type() -> type:
        """Return the SplitEnsembleClassifier class.

        Returns
        -------
        type[SplitEnsembleClassifier]
            The SplitEnsembleClassifier class.

        """
        return SplitEnsembleClassifier

    @staticmethod
    def get_test_parameters() -> dict[str, int]:
        """Return a dictionary of parameters to be used for testing.

        Returns
        -------
        dict[str, int]
            A dictionary of parameters to be used for testing.

        """
        return {"cv": 2}


if __name__ == "__main__":
    unittest.main()
