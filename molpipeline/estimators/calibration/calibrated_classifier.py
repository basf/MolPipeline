"""Methods for calibrating predicted probabilities."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
# Adapted by: MolPipeline developers

import warnings
from inspect import signature
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from loguru import logger
from sklearn.base import (
    BaseEstimator,
    _fit_context,  # noqa: PLC2701
    clone,
)
from sklearn.calibration import (
    CalibratedClassifierCV as SklearnCalibratedClassifierCV,
)
from sklearn.calibration import (
    _CalibratedClassifier,
    _fit_calibrator,  # noqa: PLC2701
)
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import LeaveOneOut, check_cv, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import (
    Bunch,
    _safe_indexing,  # noqa: PLC2701
    indexable,
)
from sklearn.utils._array_api import (
    get_namespace_and_device,  # noqa: PLC2701
    move_to,  # noqa: PLC2701
)
from sklearn.utils._response import (
    _get_response_values,  # noqa: PLC2701
    _process_predict_proba,  # noqa: PLC2701
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.metadata_routing import (
    _routing_enabled,  # noqa: PLC2701
    process_routing,
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,  # noqa: PLC2701
    _check_response_method,  # noqa: PLC2701
    _check_sample_weight,  # noqa: PLC2701
)
from typing_extensions import override


def merge_class_and_sample_weights(
    y: npt.ArrayLike,
    class_weight: dict[Any, float] | Literal["balanced"] | None,
    sample_weight: npt.ArrayLike | None,
) -> npt.ArrayLike | None:
    """Merge class weights and sample weights into a single sample weight array.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target values.

    class_weight : dict[Any, float] | 'balanced' | None, optional
        Class weights used for the calibration.
        If a dict, it must be provided in this form: ``{class_label: weight}``.
        Those weights won't be used for the underlying estimator training.
        See :term:`Glossary <class_weight>` for more details.

    sample_weight : npt.ArrayLike | None, optional
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    merged_sample_weight : ndarray of shape (n_samples,) or None
        Merged sample weights. If both `class_weight` and `sample_weight` are None,
        returns None.

    """
    if class_weight is None:
        return sample_weight

    class_weights_array = compute_sample_weight(class_weight, y)
    if sample_weight is None:
        return class_weights_array

    return class_weights_array * sample_weight


def _fit_classifier_calibrator_pair(  # noqa: PLR0917  # pylint: disable=R0914,R0917
    estimator: BaseEstimator,
    X: npt.ArrayLike,  # noqa: N803  # pylint: disable=C0103
    y: npt.ArrayLike,
    train: npt.NDArray[np.int_],
    test: npt.NDArray[np.int_],
    method: Literal["sigmoid", "isotonic", "temperature"],
    classes: npt.ArrayLike,
    xp: Any,
    class_weight: dict[Any, float] | Literal["balanced"] | None = None,
    sample_weight: npt.ArrayLike | None = None,
    fit_params: dict[str, Any] | None = None,
) -> _CalibratedClassifier:
    """Fit a classifier/calibration pair on a given train/test split.

    Adapted from scikit-learn to add support for class weights during calibration.

    Fit the classifier on the train set, compute its predictions on the test
    set and use the predictions as input to fit the calibrator along with the
    test labels.

    Parameters
    ----------
    estimator : estimator instance
        Cloned base estimator.

    X : array-like, shape (n_samples, n_features)
        Sample data.

    y : array-like, shape (n_samples,)
        Targets.

    train : ndarray, shape (n_train_indices,)
        Indices of the training subset.

    test : ndarray, shape (n_test_indices,)
        Indices of the testing subset.

    method : {'sigmoid', 'isotonic', 'temperature'}
        Method to use for calibration.

    classes : ndarray, shape (n_classes,)
        The target classes.

    xp : namespace
        Array API namespace.

    class_weight : dict or 'balanced', optional
        Class weights used for the calibration.
        If a dict, it must be provided in this form: ``{class_label: weight}``.
        Those weights won't be used for the underlying estimator training.
        See :term:`Glossary <class_weight>` for more details.

    sample_weight : array-like, default=None
        Sample weights for `X`.

    fit_params : dict, default=None
        Parameters to pass to the `fit` method of the underlying
        classifier.

    Returns
    -------
    calibrated_classifier : _CalibratedClassifier instance

    """
    fit_params_train = _check_method_params(X, params=fit_params, indices=train)
    X_train, y_train = _safe_indexing(X, train), _safe_indexing(y, train)  # noqa: N806  # pylint: disable=C0103
    X_test, y_test = _safe_indexing(X, test), _safe_indexing(y, test)  # noqa: N806  # pylint: disable=C0103

    estimator.fit(X_train, y_train, **fit_params_train)

    predictions, _ = _get_response_values(  # pylint: disable=W0632
        estimator,
        X_test,
        response_method=["decision_function", "predict_proba"],
    )
    if predictions.ndim == 1:
        # Reshape binary output from `(n_samples,)` to `(n_samples, 1)`
        predictions = predictions.reshape(-1, 1)

    if sample_weight is not None:
        # Check that the sample_weight dtype is consistent with the predictions
        # to avoid unintentional upcasts.
        sample_weight = _check_sample_weight(sample_weight, X, dtype=predictions.dtype)
        sw_test = _safe_indexing(sample_weight, test)
    else:
        sw_test = None
    sw_test = merge_class_and_sample_weights(y_test, class_weight, sw_test)
    return _fit_calibrator(
        estimator,
        predictions,
        y_test,
        classes,
        method,
        xp=xp,
        sample_weight=sw_test,
    )


class CalibratedClassifierCV(SklearnCalibratedClassifierCV):
    """Calibrate probabilities using isotonic, sigmoid, or temperature scaling.

    This class is adapted from scikit-learn to add support for class weights during
    calibration. In addtion code was adapted to adhere to local style guides.
    The plan is to upstream these changes to scikit-learn in the future, once a
    rigerous study has been performed to validate the approach.

    This class uses cross-validation to both estimate the parameters of a
    classifier and subsequently calibrate a classifier. With
    `ensemble=True`, for each cv split it
    fits a copy of the base estimator to the training subset, and calibrates it
    using the testing subset. For prediction, predicted probabilities are
    averaged across these individual calibrated classifiers. When
    `ensemble=False`, cross-validation is used to obtain unbiased predictions,
    via :func:`~sklearn.model_selection.cross_val_predict`, which are then
    used for calibration. For prediction, the base estimator, trained using all
    the data, is used. This is the prediction method implemented when
    `probabilities=True` for :class:`~sklearn.svm.SVC` and :class:`~sklearn.svm.NuSVC`
    estimators (see :ref:`User Guide <scores_probabilities>` for details).

    Already fitted classifiers can be calibrated by wrapping the model in a
    :class:`~sklearn.frozen.FrozenEstimator`. In this case all provided
    data is used for calibration. The user has to take care manually that data
    for model fitting and calibration are disjoint.

    The calibration is based on the :term:`decision_function` method of the
    `estimator` if it exists, else on :term:`predict_proba`.

    Read more in the :ref:`User Guide <calibration>`.
    In order to learn more on the CalibratedClassifierCV class, see the
    following calibration examples:
    :ref:`sphx_glr_auto_examples_calibration_plot_calibration.py`,
    :ref:`sphx_glr_auto_examples_calibration_plot_calibration_curve.py`, and
    :ref:`sphx_glr_auto_examples_calibration_plot_calibration_multiclass.py`.

    Parameters
    ----------
    estimator : estimator instance, default=None
        The classifier whose output need to be calibrated to provide more
        accurate `predict_proba` outputs. The default classifier is
        a :class:`~sklearn.svm.LinearSVC`.

        .. versionadded:: 1.2

    method : {'sigmoid', 'isotonic', 'temperature'}, default='sigmoid'
        The method to use for calibration. Can be:

        - 'sigmoid', which corresponds to Platt's method (i.e. a binary logistic
          regression model).
        - 'isotonic', which is a non-parametric approach.
        - 'temperature', temperature scaling.

        Sigmoid and isotonic calibration methods natively support only binary
        classifiers and extend to multi-class classification using a One-vs-Rest (OvR)
        strategy with post-hoc renormalization, i.e., adjusting the probabilities after
        calibration to ensure they sum up to 1.

        In contrast, temperature scaling naturally supports multi-class calibration by
        applying `softmax(classifier_logits/T)` with a value of `T` (temperature)
        that optimizes the log loss.

        For very uncalibrated classifiers on very imbalanced datasets, sigmoid
        calibration might be preferred because it fits an additional intercept
        parameter. This helps shift decision boundaries appropriately when the
        classifier being calibrated is biased towards the majority class.

        Isotonic calibration is not recommended when the number of calibration samples
        is too low ``(≪1000)`` since it then tends to overfit.

        .. versionchanged:: 1.8
           Added option 'temperature'.

    cv : int, cross-validation generator, or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`~sklearn.model_selection.KFold`
        is used.

        Refer to the :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

        Base estimator clones are fitted in parallel across cross-validation
        iterations.

        See :term:`Glossary <n_jobs>` for more details.

        .. versionadded:: 0.24

    ensemble : bool, or "auto", default="auto"
        Determines how the calibrator is fitted.

        "auto" will use `False` if the `estimator` is a
        :class:`~sklearn.frozen.FrozenEstimator`, and `True` otherwise.

        If `True`, the `estimator` is fitted using training data, and
        calibrated using testing data, for each `cv` fold. The final estimator
        is an ensemble of `n_cv` fitted classifier and calibrator pairs, where
        `n_cv` is the number of cross-validation folds. The output is the
        average predicted probabilities of all pairs.

        If `False`, `cv` is used to compute unbiased predictions, via
        :func:`~sklearn.model_selection.cross_val_predict`, which are then
        used for calibration. At prediction time, the classifier used is the
        `estimator` trained on all the data.
        Note that this method is also internally implemented  in
        :mod:`sklearn.svm` estimators with the `probabilities=True` parameter.

        .. versionadded:: 0.24

        .. versionchanged:: 1.6
            `"auto"` option is added and is the default.

    class_weight : dict or 'balanced', default=None
        Class weights used for the calibration.
        If a dict, it must be provided in this form: ``{class_label: weight}``.
        Those weights won't be used for the underlying estimator training.
        See :term:`Glossary <class_weight>` for more details.
        If both `class_weight` and `sample_weight` are provided during
        :meth:`fit`, the sample weights are multiplied by the class weights.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    calibrated_classifiers_ : list (len() equal to cv or 1 if `ensemble=False`)
        The list of classifier and calibrator pairs.

        - When `ensemble=True`, `n_cv` fitted `estimator` and calibrator pairs.
          `n_cv` is the number of cross-validation folds.
        - When `ensemble=False`, the `estimator`, fitted on all the data, and fitted
          calibrator.

        .. versionchanged:: 0.24
            Single calibrated classifier case when `ensemble=False`.

    See Also
    --------
    calibration_curve : Compute true and predicted probabilities
        for a calibration curve.

    References
    ----------
    .. [1] B. Zadrozny & C. Elkan.
       `Obtaining calibrated probability estimates from decision trees
       and naive Bayesian classifiers
       <https://cseweb.ucsd.edu/~elkan/calibrated.pdf>`_, ICML 2001.

    .. [2] B. Zadrozny & C. Elkan.
       `Transforming Classifier Scores into Accurate Multiclass
       Probability Estimates
       <https://web.archive.org/web/20060720141520id_/http://www.research.ibm.com:80/people/z/zadrozny/kdd2002-Transf.pdf>`_,
       KDD 2002.

    .. [3] J. Platt. `Probabilistic Outputs for Support Vector Machines
       and Comparisons to Regularized Likelihood Methods
       <https://www.researchgate.net/profile/John-Platt-2/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods/links/004635154cff5262d6000000/Probabilistic-Outputs-for-Support-Vector-Machines-and-Comparisons-to-Regularized-Likelihood-Methods.pdf>`_,
       1999.

    .. [4] A. Niculescu-Mizil & R. Caruana.
       `Predicting Good Probabilities with Supervised Learning
       <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>`_,
       ICML 2005.

    .. [5] Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger.
       :doi:`On Calibration of Modern Neural Networks<10.48550/arXiv.1706.04599>`.
       Proceedings of the 34th International Conference on Machine Learning,
       PMLR 70:1321-1330, 2017.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.calibration import CalibratedClassifierCV
    >>> X, y = make_classification(n_samples=100, n_features=2,
    ...                            n_redundant=0, random_state=42)
    >>> base_clf = GaussianNB()
    >>> calibrated_clf = CalibratedClassifierCV(base_clf, cv=3)
    >>> calibrated_clf.fit(X, y)
    CalibratedClassifierCV(...)
    >>> len(calibrated_clf.calibrated_classifiers_)
    3
    >>> calibrated_clf.predict_proba(X)[:5, :]
    array([[0.110, 0.889],
           [0.072, 0.927],
           [0.928, 0.072],
           [0.928, 0.072],
           [0.072, 0.928]])
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, n_features=2,
    ...                            n_redundant=0, random_state=42)
    >>> X_train, X_calib, y_train, y_calib = train_test_split(
    ...        X, y, random_state=42
    ... )
    >>> base_clf = GaussianNB()
    >>> base_clf.fit(X_train, y_train)
    GaussianNB()
    >>> from sklearn.frozen import FrozenEstimator
    >>> calibrated_clf = CalibratedClassifierCV(FrozenEstimator(base_clf))
    >>> calibrated_clf.fit(X_calib, y_calib)
    CalibratedClassifierCV(...)
    >>> len(calibrated_clf.calibrated_classifiers_)
    1
    >>> calibrated_clf.predict_proba([[-0.5, 0.5]])
    array([[0.936, 0.063]])

    """

    class_weight: dict[Any, float] | Literal["balanced"] | None

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        method: Literal["sigmoid", "isotonic", "temperature"] = "sigmoid",
        cv: Any = None,
        n_jobs: int | None = None,
        ensemble: Literal["auto"] | bool = "auto",
        class_weight: dict[Any, float] | Literal["balanced"] | None = None,
    ):
        """Initialize the CalibratedClassifierCV instance.

        Parameters
        ----------
        estimator : BaseEstimator | None, optional
            The classifier whose output need to be calibrated to provide more
            accurate `predict_proba` outputs. The default classifier is
            a :class:`~sklearn.svm.LinearSVC`.
        method : Literal['sigmoid', 'isotonic', 'temperature'], default='sigmoid'
            The method to use for calibration. Can be:
            - 'sigmoid', which corresponds to Platt's method (i.e. A binary logistic
              regression model)
            - 'isotonic', which is a non-parametric approach
            - 'temperature', temperature scaling
        cv : int, cross-validation generator, or iterable, optional
            Determines the cross-validation splitting strategy.
        n_jobs : int, optional
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            The value ``-1`` means using all processors.
        ensemble : Literal["auto"] | bool, default="auto"
            Determines how the calibrator is fitted.
            The value "auto" will use `False` if the `estimator` is a
            :class:`~sklearn.frozen.FrozenEstimator`, and `True` otherwise.
        class_weight : dict | Literal['balanced'], optional
            Class weights used for the calibration.
            If a dict, it must be provided in this form: ``{class_label: weight}``.
            Those weights won't be used for the underlying estimator training.
            See :term:`Glossary <class_weight>` for more details.

        """
        if method == "temperature" and class_weight is not None:
            logger.warning(
                "At the moment temperature scaling with class weights "
                "seems to have no effect.",
            )
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            ensemble=ensemble,
        )
        self.class_weight = class_weight

    @override
    @_fit_context(
        # CalibratedClassifierCV.estimator is not validated yet
        prefer_skip_nested_validation=False,
    )
    def fit(  # noqa: PLR0912, PLR0914, PLR0915  # pylint: disable=R0912,R0914,R0915
        self,
        X: npt.ArrayLike,  # pylint: disable=C0103
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> Self:
        """Fit the calibrated model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        **fit_params : dict
            Parameters to pass to the `fit` method of the underlying
            classifier.

        Returns
        -------
        self : object
            Returns an instance of self.

        Raises
        ------
        ValueError
            If `cv` is LeaveOneOut, because a model cannot be calibrated on a single
            sample.

        """
        check_classification_targets(y)
        X, y = indexable(X, y)  # noqa: N806
        estimator = self._get_estimator()

        ensemble = self.ensemble
        if ensemble == "auto":
            ensemble = not isinstance(estimator, FrozenEstimator)

        self.calibrated_classifiers_ = []  # pylint: disable=W0201

        # Set `classes_` using all `y`
        label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = label_encoder_.classes_  # pylint: disable=W0201
        if self.method == "temperature" and isinstance(y[0], str):  # type: ignore
            # for temperature scaling if `y` contains strings then encode it
            # right here to avoid fitting LabelEncoder again within the
            # `_fit_calibrator` function.
            y = label_encoder_.transform(y=y)

        if _routing_enabled():
            routed_params = process_routing(
                self,
                "fit",
                sample_weight=sample_weight,
                **fit_params,
            )
        else:
            # sample_weight checks
            fit_parameters = signature(estimator.fit).parameters
            supports_sw = "sample_weight" in fit_parameters
            if sample_weight is not None and not supports_sw:
                estimator_name = type(estimator).__name__
                warnings.warn(
                    f"Since {estimator_name} does not appear to accept"
                    " sample_weight, sample weights will only be used for the"
                    " calibration itself. This can be caused by a limitation of"
                    " the current scikit-learn API. See the following issue for"
                    " more details:"
                    " https://github.com/scikit-learn/scikit-learn/issues/21134."
                    " Be warned that the result of the calibration is likely to be"
                    " incorrect.",
                    stacklevel=2,
                )
            routed_params = Bunch()
            routed_params.splitter = Bunch(split={})  # no routing for splitter
            routed_params.estimator = Bunch(fit=fit_params)
            if sample_weight is not None and supports_sw:
                routed_params.estimator.fit["sample_weight"] = sample_weight

        xp, is_array_api, device_ = get_namespace_and_device(X)
        if is_array_api:
            y, sample_weight = move_to(  # type: ignore  # pylint: disable=W0632
                y,
                sample_weight,
                xp=xp,
                device=device_,
            )
        # Check that each cross-validation fold can have at least one
        # example per class
        if isinstance(self.cv, int):
            n_folds = self.cv
        elif hasattr(self.cv, "n_splits"):
            n_folds = self.cv.n_splits
        else:
            n_folds = None
        if n_folds and xp.any(xp.unique_counts(y)[1] < n_folds):
            raise ValueError(
                f"Requesting {n_folds}-fold "
                "cross-validation but provided less than "
                f"{n_folds} examples for at least one class.",
            )
        if isinstance(self.cv, LeaveOneOut):
            raise ValueError(
                "LeaveOneOut cross-validation does not allow"
                "all classes to be present in test splits. "
                "Please use a cross-validation generator that allows "
                "all classes to appear in every test and train split.",
            )
        cv = check_cv(self.cv, y, classifier=True)

        if ensemble:
            parallel = Parallel(n_jobs=self.n_jobs)
            self.calibrated_classifiers_ = parallel(  # pylint: disable=W0201
                delayed(_fit_classifier_calibrator_pair)(
                    clone(estimator),
                    X,
                    y,
                    train=train,
                    test=test,
                    method=self.method,
                    classes=self.classes_,
                    class_weight=self.class_weight,
                    sample_weight=sample_weight,
                    fit_params=routed_params.estimator.fit,
                    xp=xp,
                )
                for train, test in cv.split(X, y, **routed_params.splitter.split)
            )
        else:
            calibration_sample_weight = merge_class_and_sample_weights(
                y,
                self.class_weight,
                sample_weight,
            )
            this_estimator = clone(estimator)
            method_name = _check_response_method(
                this_estimator,
                ["decision_function", "predict_proba"],
            ).__name__
            predictions = cross_val_predict(
                estimator=this_estimator,
                X=X,
                y=y,
                cv=cv,
                method=method_name,
                n_jobs=self.n_jobs,
                params=routed_params.estimator.fit,
            )
            if self.classes_.shape[0] == 2:  # noqa: PLR2004
                # Ensure shape (n_samples, 1) in the binary case
                if method_name == "predict_proba":
                    # Select the probability column of the positive class
                    predictions = _process_predict_proba(
                        y_pred=predictions,
                        target_type="binary",
                        classes=self.classes_,
                        pos_label=self.classes_[1],
                    )
                predictions = predictions.reshape(-1, 1)

            if calibration_sample_weight is not None:
                # Check that the sample_weight dtype is consistent with the
                # predictions to avoid unintentional upcasts.
                calibration_sample_weight = _check_sample_weight(
                    calibration_sample_weight,
                    predictions,
                    dtype=predictions.dtype,
                )

            this_estimator.fit(X, y, **routed_params.estimator.fit)
            # Note: Here we don't pass on fit_params because the supported
            # calibrators don't support fit_params anyway
            calibrated_classifier = _fit_calibrator(
                this_estimator,
                predictions,
                y,
                self.classes_,
                self.method,
                sample_weight=calibration_sample_weight,
                xp=xp,
            )
            self.calibrated_classifiers_.append(calibrated_classifier)

        first_clf = self.calibrated_classifiers_[0].estimator
        if hasattr(first_clf, "n_features_in_"):
            self.n_features_in_ = first_clf.n_features_in_  # pylint: disable=W0201
        if hasattr(first_clf, "feature_names_in_"):
            self.feature_names_in_ = first_clf.feature_names_in_  # pylint: disable=W0201
        return self
