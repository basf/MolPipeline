"""Utilities for randomization ensuring that tests do not rely on specific values."""

import numpy as np
import torch


def randomize_state_dict_weights(
    state_dict: dict[str, torch.Tensor],
    random_state: int | None = None,
) -> dict[str, torch.Tensor]:
    """Randomize the weights in a state_dict.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The state_dict to randomize.
    random_state : int | None, default=None
        The random state to use for reproducibility.

    Returns
    -------
    dict[str, torch.Tensor]
        The randomized state_dict.

    Raises
    ------
    ValueError
        If the state dict does not contain any weights to randomize.

    """
    random_state_dict = {}
    rng = np.random.default_rng(random_state)
    altered_keys = []
    for key, value in state_dict.items():
        if key.endswith(".weight"):
            altered_keys.append(key)
            random_weights = rng.normal(size=value.shape).astype(np.float32)
            random_state_dict[key] = torch.tensor(random_weights)
        else:
            random_state_dict[key] = value
    if not altered_keys:
        raise ValueError("No weights were altered in the state_dict.")
    return random_state_dict
