"""Utility functions for visualization of molecules and their explanations."""

import io
from typing import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.pyplot import get_cmap
from PIL import Image
from rdkit import Chem

# red green blue alpha tuple
RGBAtuple = tuple[float, float, float, float]


def get_mol_lims(mol: Chem.Mol) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the extent of the molecule.

    x- and y-coordinates of all atoms in the molecule are accessed, returning min- and max-values for both axes.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit Molecule object of which the limits are determined.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        Limits of the molecule.
    """
    coords_list = []
    conf = mol.GetConformer(0)
    for i, _ in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords_list.append((pos.x, pos.y))
    coords: npt.NDArray[np.float64] = np.array(coords_list)
    min_p = np.min(coords, axis=0)
    max_p = np.max(coords, axis=0)
    x_lim = min_p[0], max_p[0]
    y_lim = min_p[1], max_p[1]
    return x_lim, y_lim


def pad(
    lim: Sequence[float] | npt.NDArray[np.float64], ratio: float
) -> tuple[float, float]:
    """Take a 2-dimensional vector and adds len(vector) * ratio / 2 to each side and returns obtained vector.

    Parameters
    ----------
    lim: Sequence[float] | npt.NDArray[np.float64]
        Limits which are extended.
    ratio: float
        factor by which the limits are extended.

    Returns
    -------
    List[float, float]
        Extended limits
    """
    diff = max(lim) - min(lim)
    diff *= ratio / 2
    return lim[0] - diff, lim[1] + diff


def get_color_map_from_input(
    color: str | Colormap | tuple[RGBAtuple, RGBAtuple, RGBAtuple] | None
) -> Colormap:
    """Get a colormap from a user defined color scheme.

    Parameters
    ----------
    color: str | Colormap | tuple[RGBAtuple, RGBAtuple, RGBAtuple] | None
        The color scheme.

    Returns
    -------
    Colormap
        The colormap.
    """
    # read user definer color scheme as ColorMap
    if color is None:
        coolwarm = (
            (0.017, 0.50, 0.850, 0.5),
            (1.0, 1.0, 1.0, 0.5),
            (1.0, 0.25, 0.0, 0.5),
        )
        coolwarm = (coolwarm[2], coolwarm[1], coolwarm[0])
        color = coolwarm
    if isinstance(color, Colormap):
        color_map = color
    elif isinstance(color, tuple):
        color_map = color_tuple_to_colormap(color)  # type: ignore
    elif isinstance(color, str):
        color_map = get_cmap(color)
    else:
        raise ValueError("Color must be a tuple, string or ColorMap.")
    return color_map


def color_tuple_to_colormap(
    color_tuple: tuple[RGBAtuple, RGBAtuple, RGBAtuple]
) -> Colormap:
    """Convert a color tuple to a colormap.

    Parameters
    ----------
    color_tuple: tuple[RGBAtuple, RGBAtuple, RGBAtuple]
        The color tuple.

    Returns
    -------
    Colormap
        The colormap (a matplotlib data structure).
    """
    if len(color_tuple) != 3:
        raise ValueError("Color tuple must have 3 elements")

    # Definition of color
    col1, col2, col3 = map(np.array, color_tuple)

    # Creating linear gradient for color mixing
    linspace = np.linspace(0, 1, int(128))
    linspace4d = np.vstack([linspace] * 4).T

    # interpolating values for 0 to 0.5 by mixing purple and white
    zero_to_half = linspace4d * col2 + (1 - linspace4d) * col3
    # interpolating values for 0.5 to 1 by mixing white and yellow
    half_to_one = col1 * linspace4d + col2 * (1 - linspace4d)

    # Creating new colormap from
    color_map = ListedColormap(np.vstack([zero_to_half, half_to_one]))
    return color_map


def to_png(data: bytes) -> Image.Image:
    """Show a PNG image from a byte stream.

    Parameters
    ----------
    data: bytes
        The image data.

    Returns
    -------
    Image
        The image.
    """
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img


def plt_to_pil(figure: plt.Figure) -> Image.Image:
    """Convert a matplotlib figure to a PIL image.

    Parameters
    ----------
    figure: plt.Figure
        The figure.

    Returns
    -------
    Image
        The image.
    """
    bio = io.BytesIO()
    figure.savefig(bio, format="png")
    bio.seek(0)
    img = Image.open(bio)
    return img
