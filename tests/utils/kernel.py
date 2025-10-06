"""Legacy import path for kernel functions."""

from loguru import logger

from molpipeline.kernel.tanimoto_functions import (
    self_tanimoto_distance,
    self_tanimoto_similarity,
    tanimoto_distance_sparse,
    tanimoto_similarity_sparse,
)

logger.warning(
    "The kernel functions have been moved to molpipeline.kernel.tanimoto_functions and "
    "will be removed from utils.kernel in future releases.",
)

__all__ = [
    "self_tanimoto_distance",
    "self_tanimoto_similarity",
    "tanimoto_distance_sparse",
    "tanimoto_similarity_sparse",
]
