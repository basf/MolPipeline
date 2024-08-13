"""Module for generating heatmaps from 2D-grids.

Much of the visualization code in this file originates from projects of Christian W. Feldmann:
    https://github.com/c-feldmann/rdkit_heatmaps
    https://github.com/c-feldmann/compchemkit
"""

import abc
from typing import Sequence, Callable

import numpy as np
import numpy.typing as npt
from matplotlib import cm, colors
from rdkit.Chem import Draw
from rdkit.Geometry.rdGeometry import Point2D


class Grid2D(abc.ABC):
    """Metaclass for discrete 2-dimensional grids.

    This class holds a matrix of values accessed by index, where each cell is associated with a specific location.
    """

    def __init__(
        self,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
        x_res: int,
        y_res: int,
    ) -> None:
        """Initializes the Grid2D with limits and resolution of the axes.

        Parameters
        ----------
        x_lim: Tuple[float, float]
            Extend of the grid along the x-axis (xmin, xmax).
        y_lim: Tuple[float, float]
            Extend of the grid along the y-axis (ymin, ymax).
        x_res: int
            Resolution (number of cells) along x-axis.
        y_res: int
            Resolution (number of cells) along y-axis.
        """
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_res = x_res
        self.y_res = y_res
        self.values = np.zeros((self.x_res, self.y_res))

    @property
    def dx(self) -> float:
        """Length of cell in x-direction."""
        return (max(self.x_lim) - min(self.x_lim)) / self.x_res

    @property
    def dy(self) -> float:
        """Length of cell in y-direction."""
        return (max(self.y_lim) - min(self.y_lim)) / self.y_res

    def grid_field_center(self, x_idx: int, y_idx: int) -> tuple[float, float]:
        """Center of cell specified by index along x and y.

        Parameters
        ----------
        x_idx: int
             cell-index along x-axis.
        y_idx:int
             cell-index along y-axis.

        Returns
        -------
        Tuple[float, float]
            Coordinates of center of cell.
        """
        x_coord = min(self.x_lim) + self.dx * (x_idx + 0.5)
        y_coord = min(self.y_lim) + self.dy * (y_idx + 0.5)
        return x_coord, y_coord

    def grid_field_lim(
        self, x_idx: int, y_idx: int
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get x and y coordinates for the upper left and lower right position of specified pixel.

        x_idx: int
            cell-index along x-axis.
        y_idx: int
            cell-index along y-axis.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            Coordinates of upper left and lower right corner of cell.
        """
        upper_left = (
            min(self.x_lim) + self.dx * x_idx,
            min(self.y_lim) + self.dy * y_idx,
        )
        lower_right = (
            min(self.x_lim) + self.dx * (x_idx + 1),
            min(self.y_lim) + self.dy * (y_idx + 1),
        )
        return upper_left, lower_right


class ColorGrid(Grid2D):
    """Stores rgba-values of cells."""

    def __init__(
        self,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
        x_res: int,
        y_res: int,
    ):
        """Initialize the ColorGrid with limits and resolution of the axes.

        x_lim: tuple[float, float]
            Extend of the grid along the x-axis (xmin, xmax).
        y_lim: tuple[float, float]
            Extend of the grid along the y-axis (ymin, ymax).
        x_res: int
            Resolution (number of cells) along x-axis.
        y_res: int
            Resolution (number of cells) along y-axis.
        """
        super().__init__(x_lim, y_lim, x_res, y_res)
        self.color_grid = np.ones((self.x_res, self.y_res, 4))


class ValueGrid(Grid2D):
    """Calculate and store values of cells.

    Evaluates all added functions for the position of each cell and calculates the value of each cell as sum of these
    functions.
    """

    def __init__(
        self,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
        x_res: int,
        y_res: int,
    ):
        """Initialize the ValueGrid with limits and resolution of the axes.

        Parameters
        ----------
        x_lim: Tuple[float, float]
            Extend of the grid along the x-axis (xmin, xmax).
        y_lim: Tuple[float, float]
            Extend of the grid along the y-axis (ymin, ymax).
        x_res: int
            Resolution (number of cells) along x-axis.
        y_res: int
            Resolution (number of cells) along y-axis.
        """
        super().__init__(x_lim, y_lim, x_res, y_res)
        self.function_list: list[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ] = []
        self.values = np.zeros((self.x_res, self.y_res))

    def add_function(
        self, function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    ) -> None:
        """Add a function to the grid which is evaluated for each cell, when `self.evaluate` is called.

        Parameters
        ----------
        function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
            Function to be evaluated for each cell. The function should take an array of positions and return an array
            of values, e.g. a Gaussian function.
        """
        self.function_list.append(function)

    def evaluate(self) -> None:
        """Evaluate each function for each cell. Values of cells are calculated as the sum of all function-values.

        The results are saved to `self.values`.
        """
        self.values = np.zeros((self.x_res, self.y_res))
        x_y0_list = np.array(
            [self.grid_field_center(x, 0)[0] for x in range(self.x_res)]
        )
        x0_y_list = np.array(
            [self.grid_field_center(0, y)[1] for y in range(self.y_res)]
        )
        xv, yv = np.meshgrid(x_y0_list, x0_y_list)
        xv = xv.ravel()
        yv = yv.ravel()
        coordinate_pairs = np.vstack([xv, yv]).T
        for f in self.function_list:
            values = f(coordinate_pairs)
            values = values.reshape(self.y_res, self.x_res).T
            assert values.shape == self.values.shape, (values.shape, self.values.shape)
            self.values += values

    def map2color(
        self,
        c_map: colors.Colormap | str,
        v_lim: Sequence[float] | None = None,
    ) -> ColorGrid:
        """Generate a ColorGrid from `self.values` according to given colormap

        Parameters
        ----------
        c_map: Union[colors.Colormap, str]
            Colormap to be used for mapping values to colors.
        v_lim: Optional[Tuple[float, float]]
            Limits for the colormap. If not given, the maximum absolute value of `self.values` is used as limit.

        Returns
        -------
        ColorGrid
            ColorGrid with colors corresponding to ValueGrid.
        """
        color_grid = ColorGrid(self.x_lim, self.y_lim, self.x_res, self.y_res)
        if not v_lim:
            abs_max = np.max(np.abs(self.values))
            v_lim = -abs_max, abs_max
        normalizer = colors.Normalize(vmin=v_lim[0], vmax=v_lim[1])
        if isinstance(c_map, str):
            c_map = cm.get_cmap(c_map)
        norm = normalizer(self.values)
        color_grid.color_grid = np.array(c_map(norm))
        return color_grid


def color_canvas(canvas: Draw.MolDraw2D, color_grid: ColorGrid) -> None:
    """Draw a ColorGrid object to a RDKit Draw.MolDraw2D canvas.

    Each pixel is drawn as rectangle, so if you use Draw.MolDrawSVG brace yourself and your RAM!

    Parameters
    ----------
    canvas: Draw.MolDraw2D
        RDKit Draw.MolDraw2D canvas.
    color_grid: ColorGrid
        ColorGrid object to be drawn on the canvas.
    """
    for x in range(color_grid.x_res):
        for y in range(color_grid.y_res):
            upper_left, lower_right = color_grid.grid_field_lim(x, y)
            upper_left, lower_right = Point2D(*upper_left), Point2D(*lower_right)
            canvas.SetColour(tuple(color_grid.color_grid[x, y]))
            canvas.DrawRect(upper_left, lower_right)
