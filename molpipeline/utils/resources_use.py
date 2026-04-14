"""Change resources of estimators."""

import importlib.util
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

try:
    from typing import TypeIs
except ImportError:
    from typing_extensions import TypeIs
from unittest.mock import MagicMock

from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from molpipeline.utils.molpipeline_types import AnySklearnEstimator

_CHEMPROP_INSTALLED = importlib.util.find_spec("chemprop") is not None

if _CHEMPROP_INSTALLED or TYPE_CHECKING:
    from molpipeline.estimators.chemprop.abstract import ABCChemprop


def _is_chemprop(component: Any) -> TypeIs[ABCChemprop]:  # pylint: disable=possibly-used-before-assignment
    """Check if a component is an ABCChemprop instance.

    Returns ``False`` when chemprop is not installed, avoiding a
    ``NameError`` on the conditionally-imported ``ABCChemprop`` symbol.

    Parameters
    ----------
    component : Any
        The component to check.

    Returns
    -------
    bool
        ``True`` if *component* is an ``ABCChemprop`` instance.

    """
    return _CHEMPROP_INSTALLED and isinstance(component, ABCChemprop)  # pylint: disable=possibly-used-before-assignment


def iterate_components(
    estimator: Any,
    prefix: str = "",
    seen: set[int] | None = None,
) -> Generator[tuple[Any, str], None, None]:
    """Recursively iterate over all components in an estimator.

    Parameters
    ----------
    estimator : Any
        The estimator to iterate over. Accepts any object, including
        sklearn-compatible estimators. MagicMock objects are returned early
        to avoid infinite attribute recursion.
    prefix : str, default=""
        The prefix to add to the component names. Used for logging purposes.
    seen : set[int] | None, optional
        Set of ``id()`` values of already-visited estimator objects. Used to
        detect circular references and abort recursion. If None, a new set is created.

    Yields
    ------
    Any
        Each component estimator found in the estimator tree.
    str
        The name of the component, prefixed by the names of parent components.

    """
    if seen is None:
        seen = set()

    if isinstance(estimator, MagicMock):
        # guard to avoid endless recursions from MagicMock which might create attributes
        # dynamically when they are checked with `hasattr`.
        # Checking for public types like sklearn's BaseEstimator and MolPipeline's
        # ABCPipelineElement doesn't work in all cases. For example,
        # CalibratedClassifierCV a private wrapper `_CalibratedClassifier` that is not
        # typed with any public interface.
        return

    estimator_id = id(estimator)
    if estimator_id in seen:
        logger.debug(
            f"Skipping already-visited {estimator.__class__.__name__} "
            f"(id={estimator_id}) to avoid circular recursion.",
        )
        return
    seen.add(estimator_id)
    yield estimator, prefix

    # block that traverses nested estimator objects
    if isinstance(estimator, Pipeline):
        for step in estimator.steps:
            yield from iterate_components(step[1], f"{prefix}{step[0]}__", seen)
    if hasattr(estimator, "estimator"):  # type: ignore
        # follow nested estimators in wrapping estimators like CalibratedClassifierCV,
        # RandomForestClassifier. `estimator` in sklearn and MolPipeline is usually
        # stored as the original estimator which is cloned for an ensemble.
        yield from iterate_components(estimator.estimator, f"{prefix}estimator__", seen)  # type: ignore

    if hasattr(estimator, "estimators") and isinstance(estimator.estimators, list):  # type: ignore
        # VotingClassifier and VotingRegressor do not use estimator, but estimators.
        for name, est in estimator.estimators:  # type: ignore
            yield from iterate_components(est, f"{prefix}{name}__", seen)

    if hasattr(estimator, "estimators_") and isinstance(estimator.estimators_, list):  # type: ignore
        # follow nested estimators in wrapping estimators like RandomForestClassifier.
        # `estimators_` in sklearn and MolPipeline is usually the list of cloned
        # estimators for an ensemble which are only available after fitting. These are
        # usually used for making predictions.
        for i, est in enumerate(estimator.estimators_):  # type: ignore
            yield from iterate_components(est, f"{prefix}estimator_{i}__", seen)

    if isinstance(estimator, CalibratedClassifierCV) and hasattr(
        estimator,
        "calibrated_classifiers_",
    ):
        # CalibratedClassifierCV doesn't store `estimators_` but
        # `calibrated_classifiers_`.
        for i, est in enumerate(estimator.calibrated_classifiers_):
            yield from iterate_components(
                est,
                f"{prefix}calibrated_classifiers_{i}__",
                seen,
            )


def model_to_cpu(model: AnySklearnEstimator | None) -> bool:
    """Change the device of a chempropModel to CPU.

    Changes are made in place.

    Parameters
    ----------
    model : AnySklearnEstimator | None
        The model to transfer to CPU.
        None is accepted and results in no change (returns False).

    Returns
    -------
    bool
        True if the model was changed, False otherwise.

    """
    if model is None:
        return False
    changed = False
    for component, name in iterate_components(model):
        if _is_chemprop(component):
            accelerator = component.get_params()["lightning_trainer__accelerator"]
            if accelerator != "cpu":
                component.set_params(lightning_trainer__accelerator="cpu")
                logger.info(
                    f"Setting {name}lightning_trainer__accelerator "
                    f"({component.__class__.__name__}) from {accelerator} to cpu.",
                )
                changed = True
    return changed


def set_n_job_estimator(
    estimator: AnySklearnEstimator,
    n_jobs: int,
    n_jobs_chemprop: int | None,
) -> bool:
    """Recursively set ``n_jobs`` on an estimator and its nested components.

    Parameters
    ----------
    estimator : AnySklearnEstimator
        The sklearn-compatible estimator to update.
    n_jobs : int
        Target ``n_jobs`` value for non-chemprop estimators.
    n_jobs_chemprop : int | None
        Target ``n_jobs`` value for chemprop estimators. Falls back to
        ``n_jobs`` when ``None``.

    Returns
    -------
    bool
        ``True`` if any ``n_jobs`` value was changed, ``False`` otherwise.

    """
    changed = False
    for component, name in iterate_components(estimator):
        if hasattr(component, "n_jobs"):
            n_jobs_to_set = n_jobs
            if _is_chemprop(component) and n_jobs_chemprop is not None:
                n_jobs_to_set = n_jobs_chemprop
            if component.n_jobs != n_jobs_to_set:
                logger.info(
                    f"Setting {name}n_jobs ({component.__class__.__name__})"
                    f" from {component.n_jobs} to {n_jobs_to_set}.",
                )
                component.n_jobs = n_jobs_to_set
                changed = True
    return changed


def set_single_job(model: AnySklearnEstimator | None) -> bool:
    """Set the model to single job mode.

    Changes are made in place.

    Parameters
    ----------
    model : AnySklearnEstimator | None
        The model to set to single job mode.
        None is accepted and returns False without any changes.

    Returns
    -------
    bool
        True if the model was changed, False otherwise.

    Notes
    -----
    For chemprop n_jobs defines the number of workers for the model.
    To run the model in single job mode, n_jobs must be set to 0.

    """
    if model is None:
        return False
    return set_n_job_estimator(model, n_jobs=1, n_jobs_chemprop=0)
