"""Conformal prediction utils."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy.typing as npt
from crepes.extras import hinge, margin


class NonconformityFunction(ABC):
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


class HingeNonconformity(NonconformityFunction):
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


class MarginNonconformity(NonconformityFunction):
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


class CustomNonconformity(NonconformityFunction):
    """Wrapper for custom nonconformity functions."""

    def __init__(self, func: Callable[..., npt.NDArray[Any]], name: str = "custom"):
        """Initialize with a custom function.

        Parameters
        ----------
        func : Callable[..., npt.NDArray[Any]]
            The nonconformity function.
        name : str, optional
            Name for the function, by default "custom".

        """
        self.func = func
        self.name = name

    def __call__(
        self,
        x_prob: npt.NDArray[Any],
        classes: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Compute nonconformity scores using the custom function.

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
        return self.func(x_prob, classes, y)

    def get_name(self) -> str:
        """Get function name for parameter serialization.

        Returns
        -------
        str
            Name of the nonconformity function.

        """
        return f"custom_{self.name}"


# Registry for built-in nonconformity functions
NONCONFORMITY_REGISTRY = {
    "hinge": HingeNonconformity,
    "margin": MarginNonconformity,
}


def create_nonconformity_function(
    nonconformity: (
        str
        | NonconformityFunction
        | Callable[
            [npt.NDArray[Any], npt.NDArray[Any] | None, npt.NDArray[Any] | None],
            npt.NDArray[Any],
        ]
        | None
    ),
    name: str = "custom",
) -> NonconformityFunction | None:
    """Create a nonconformity function wrapper.

    Parameters
    ----------
    nonconformity : str | NonconformityFunction | Callable | None
        The nonconformity specification.
    name : str, optional
        Name for custom functions, by default "custom".

    Returns
    -------
    NonconformityFunction | None
        The wrapped nonconformity function, or None if input is None.

    Raises
    ------
    ValueError
        If the nonconformity specification is invalid.

    """
    if nonconformity is None:
        return None

    if isinstance(nonconformity, NonconformityFunction):
        return nonconformity

    if isinstance(nonconformity, str):
        if nonconformity in NONCONFORMITY_REGISTRY:
            return NONCONFORMITY_REGISTRY[nonconformity]()
        raise ValueError(
            f"Unknown nonconformity function '{nonconformity}'. "
            f"Available options: {list(NONCONFORMITY_REGISTRY.keys())}",
        )

    if callable(nonconformity):
        return CustomNonconformity(nonconformity, name)

    raise ValueError(
        f"Invalid nonconformity specification: {type(nonconformity)}. "
        "Expected str, NonconformityFunction, callable, or None.",
    )


def resolve_nonconformity_from_name(name: str | None) -> NonconformityFunction | None:
    """Resolve nonconformity function from name (for parameter serialization).

    Parameters
    ----------
    name : str | None
        Name of the nonconformity function.

    Returns
    -------
    NonconformityFunction | None
        The nonconformity function or None.

    Raises
    ------
    ValueError
        If the function name is unknown or unsupported.

    """
    if name is None:
        return None
    if name in NONCONFORMITY_REGISTRY:
        return NONCONFORMITY_REGISTRY[name]()
    if name.startswith("custom_"):
        # Custom functions can't be restored from name only
        raise ValueError(
            f"Cannot restore custom function from name: {name}. "
            "Custom functions require the original function object.",
        )
    raise ValueError(f"Unknown nonconformity function name: {name}")
