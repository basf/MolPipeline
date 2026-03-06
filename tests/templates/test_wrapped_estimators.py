"""Template tests for wrapped estimators."""

import abc

from tests.utils.mock_estimators import (
    MockEstimator,
)


class WrappedEstimatorBaseTestMixIn(abc.ABC):
    """Unit tests for CloneEnsembleRegressor."""

    @staticmethod
    @abc.abstractmethod
    def get_wrapped_estimator_type() -> type:
        """Construct the wrapped estimator to be tested.

        Returns
        -------
        type
            The class of the wrapped estimator to be tested.

        """

    def test_param_forwarding(self) -> None:
        """Parameters are forwarded to the wrapped estimator.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        base = MockEstimator(alpha=1)
        estimator_class = self.get_wrapped_estimator_type()
        ensemble = estimator_class(
            estimator=base,
            estimator__beta=2,
        )
        ensemble.set_params(estimator__gamma=3)
        base_est = ensemble.estimator
        if not isinstance(base_est, MockEstimator):
            raise TypeError("Expected an instance of MockEstimator")
        self.assertEqual(base_est.alpha, 1)  # type: ignore
        self.assertEqual(base_est.beta, 2)  # type: ignore
        self.assertEqual(base_est.gamma, 3)  # type: ignore

    def test_get_params(self) -> None:
        """get_params exposes nested estimator parameters."""
        base = MockEstimator(alpha=1)
        estimator_class = self.get_wrapped_estimator_type()
        ensemble = estimator_class(
            estimator=base,
            estimator__beta=2,
        )
        ensemble.set_params(estimator__gamma=3)
        params = ensemble.get_params(deep=True)
        self.assertIn("estimator__alpha", params)  # type: ignore
        self.assertIn("estimator__beta", params)  # type: ignore
        self.assertIn("estimator__gamma", params)  # type: ignore
        self.assertEqual(params["estimator__alpha"], 1)  # type: ignore
        self.assertEqual(params["estimator__beta"], 2)  # type: ignore
        self.assertEqual(params["estimator__gamma"], 3)  # type: ignore
