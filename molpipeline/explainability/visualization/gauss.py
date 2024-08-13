import numpy as np
import numpy.typing as npt


class GaussFunction2D:
    def __init__(self, center, std1=1, std2=1, scale=1, rotation=0):
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

    def __call__(self, pos: npt.NDArray):
        exponent = self._a * (pos[:, 0] - self.center[0]) ** 2
        exponent += (
            2 * self._b * (pos[:, 0] - self.center[0]) * (pos[:, 1] - self.center[1])
        )
        exponent += self._c * (pos[:, 1] - self.center[1]) ** 2
        return self.scale * np.exp(-exponent)
