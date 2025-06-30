"""Utils for explainability tests."""

from typing import Any

import scipy

from molpipeline import Pipeline
from molpipeline.utils.subpipeline import get_featurization_subpipeline


def construct_kernel_shap_kwargs(pipeline: Pipeline, data: list[str]) -> dict[str, Any]:
    """Construct the kwargs for SHAPKernelExplainer.

    Convert sparse matrix to dense array because SHAPKernelExplainer does not
    support sparse matrix as `data` and then explain dense matrices.
    We stick to dense matrices for simplicity.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline used for featurization.
    data : list[str]
        The input data, e.g. SMILES strings.

    Returns
    -------
    dict[str, Any]
        The kwargs for SHAPKernelExplainer
    """
    featurization_subpipeline = get_featurization_subpipeline(
        pipeline, raise_not_found=True
    )
    data_transformed = featurization_subpipeline.transform(data)  # type: ignore[union-attr]
    if scipy.sparse.issparse(data_transformed):
        data_transformed = data_transformed.toarray()
    return {"data": data_transformed}
