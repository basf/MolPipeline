"""Conformal prediction utils."""

from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt
from crepes.extras import hinge, margin


class NonconformityFunctor(ABC):
    """Abstract base class for nonconformity functions.

    Simple wrapper for nonconformity functions with sklearn compatibility.
    Supports pickle serialization and parameter management.
    """

    @abstractmethod
    def __call__(
        self,
        x_prob: npt.NDArray[Any],
        classes: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Compute nonconformity scores.

        Parameters
        ----------
        x_prob : npt.NDArray[Any]
            Probability predictions or raw predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[Any]
            Nonconformity scores.

        """

    @abstractmethod
    def get_name(self) -> str:
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """


class HingeNonconformity(NonconformityFunctor):
    """Hinge loss nonconformity function for classification."""

    def __call__(
        self,
        x_prob: npt.NDArray[Any],
        classes: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Compute hinge nonconformity scores.

        Parameters
        ----------
        x_prob : npt.NDArray[Any]
            Probability predictions or raw predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[Any]
            Nonconformity scores.

        """
        return hinge(x_prob, classes, y)

    def get_name(self) -> str:  # noqa: PLR6301
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """
        return "hinge"


class MarginNonconformity(NonconformityFunctor):
    """Margin nonconformity function for classification."""

    def __call__(
        self,
        x_prob: npt.NDArray[Any],
        classes: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Compute margin nonconformity scores.

        Parameters
        ----------
        x_prob : npt.NDArray[Any]
            Probability predictions or raw predictions.
        classes : npt.NDArray[Any] | None, optional
            Class labels, by default None.
        y : npt.NDArray[Any] | None, optional
            True labels, by default None.

        Returns
        -------
        npt.NDArray[Any]
            Nonconformity scores.

        """
        return margin(x_prob, classes, y)

    def get_name(self) -> str:  # noqa: PLR6301
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """
        return "margin"


# Registry for built-in nonconformity functions
NONCONFORMITY_REGISTRY = {
    "hinge": HingeNonconformity,
    "margin": MarginNonconformity,
}


def create_nonconformity_function(
    nonconformity: str | NonconformityFunctor | None,
) -> NonconformityFunctor | None:
    """Create a nonconformity function wrapper.

    Parameters
    ----------
    nonconformity : str | NonconformityFunctor | None
        The nonconformity specification.

    Returns
    -------
    NonconformityFunctor | None
        The wrapped nonconformity function, or None if input is None.

    Raises
    ------
    ValueError
        If the nonconformity specification is invalid.

    """
    if nonconformity is None:
        return None

    if isinstance(nonconformity, NonconformityFunctor):
        return nonconformity

    if isinstance(nonconformity, str):
        if nonconformity in NONCONFORMITY_REGISTRY:
            return NONCONFORMITY_REGISTRY[nonconformity]()
        if nonconformity.startswith("custom_"):
            raise ValueError(
                f"Cannot restore custom function from name: {nonconformity}. "
                "Custom functions require the original function object.",
            )
        raise ValueError(
            f"Unknown nonconformity function '{nonconformity}'. "
            f"Available options: {list(NONCONFORMITY_REGISTRY.keys())}",
        )

    raise ValueError(
        f"Invalid nonconformity specification: {type(nonconformity)}. "
        "Expected str, NonconformityFunctor, or None.",
    )
