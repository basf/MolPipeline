"""Gaussian functions for visualization.

Much of the visualization code in this file originates from projects of Christian W. Feldmann:
    https://github.com/c-feldmann/rdkit_heatmaps
    https://github.com/c-feldmann/compchemkit
"""

import numpy as np
import numpy.typing as npt


class GaussFunction2D:
    """2D Gaussian function."""

    def __init__(
        self,
        center: tuple[float, float],
        std1: float = 1,
        std2: float = 1,
        scale: float = 1,
        rotation: float = 0,
    ) -> None:
        """Initialize 2D Gaussian function.

        center : tuple[float, float]
            Center of the Gaussian function.
        std1 : float
            Standard deviation along the first axis.
        std2 : float
            Standard deviation along the second axis.
        scale : float
            Scaling factor.
        rotation : float
            Rotation angle in radians.
        """
        self.center = np.array(center)
        self.std1 = std1
        self.std2 = std2
        self.scale = scale
        self.rotation = rotation

        self._a = np.cos(self.rotation) ** 2 / (2 * self.std1**2) + np.sin(
            self.rotation
        ) ** 2 / (2 * self.std2**2)
        self._b = -np.sin(2 * self.rotation) / (4 * self.std1**2) + np.sin(
            2 * self.rotation
        ) / (4 * self.std2**2)
        self._c = np.sin(self.rotation) ** 2 / (2 * self.std1**2) + np.cos(
            self.rotation
        ) ** 2 / (2 * self.std2**2)

    def __call__(self, pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the Gaussian function at the given positions.

        pos : npt.NDArray[np.float64]
            Array of positions to evaluate the Gaussian function at.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of function values at the given positions.
        """
        exponent = self._a * (pos[:, 0] - self.center[0]) ** 2
        exponent += (
            2 * self._b * (pos[:, 0] - self.center[0]) * (pos[:, 1] - self.center[1])
        )
        exponent += self._c * (pos[:, 1] - self.center[1]) ** 2
        return self.scale * np.exp(-exponent)
