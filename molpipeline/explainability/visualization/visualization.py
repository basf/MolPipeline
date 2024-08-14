"""Visualization functions for the explainability module.

Much of the visualization code in this file originates from projects of Christian W. Feldmann:
    https://github.com/c-feldmann/rdkit_heatmaps
    https://github.com/c-feldmann/compchemkit
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np
import numpy.typing as npt
from PIL import Image
from matplotlib.colors import Colormap, ListedColormap
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from molpipeline.abstract_pipeline_elements.core import RDKitMol
from molpipeline.explainability.visualization.gauss import GaussFunctor2D
from molpipeline.explainability.visualization.heatmaps import (
    color_canvas,
    ValueGrid,
)

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
    newcmp = ListedColormap(np.vstack([zero_to_half, half_to_one]))
    return newcmp


def _make_grid(
    mol: Chem.Mol,
    canvas: rdMolDraw2D.MolDraw2D,
    grid_resolution: Sequence[int],
    padding: Sequence[float],
) -> ValueGrid:
    """Create a grid for the molecule.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule object.
    canvas: rdMolDraw2D.MolDraw2D
        RDKit canvas.
    grid_resolution: Sequence[int]
        Resolution of the grid.
    padding: Sequence[float]
        Padding of the grid.

    Returns
    -------
    ValueGrid
        ValueGrid object.
    """
    xl: list[float]
    yl: list[float]
    xl, yl = [list(lim) for lim in get_mol_lims(mol)]  # Limit of molecule

    # Extent of the canvas is approximated by size of molecule scaled by ratio of canvas height and width.
    # Would be nice if this was directly accessible...
    mol_height = yl[1] - yl[0]
    mol_width = xl[1] - xl[0]

    height_to_width_ratio_mol = mol_height / mol_width
    height_to_width_ratio_canvas = canvas.Height() / canvas.Width()

    if height_to_width_ratio_mol < height_to_width_ratio_canvas:
        mol_height_new = canvas.Height() / canvas.Width() * mol_width
        yl[0] -= (mol_height_new - mol_height) / 2
        yl[1] += (mol_height_new - mol_height) / 2
    else:
        mol_width_new = canvas.Width() / canvas.Height() * mol_height
        xl[0] -= (mol_width_new - mol_width) / 2
        xl[1] += (mol_width_new - mol_width) / 2

    xl = list(pad(xl, padding[0]))  # Increasing size of x-axis
    yl = list(pad(yl, padding[1]))  # Increasing size of y-axis
    v_map = ValueGrid(xl, yl, grid_resolution[0], grid_resolution[1])
    return v_map


def _add_gaussians_for_atoms(
    mol: Chem.Mol,
    conf: Chem.Conformer,
    v_map: ValueGrid,
    atom_weights: npt.NDArray[np.float64],
    atom_width: float,
) -> ValueGrid:
    """Add Gauss-functions centered at atoms to the grid.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule object.
    conf: Chem.Conformer
        Conformation of the molecule.
    v_map: ValueGrid
        ValueGrid object to which the functions are added.
    atom_weights: npt.NDArray[np.float64]
        Array of weights for atoms.
    atom_width: float
        Width of the displayed atom weights.

    Returns
    -------
    ValueGrid
        ValueGrid object with added functions.
    """
    for i, _ in enumerate(mol.GetAtoms()):
        if atom_weights[i] == 0:
            continue
        pos = conf.GetAtomPosition(i)
        coords = np.array([pos.x, pos.y])
        func = GaussFunctor2D(
            center=coords,
            std1=atom_width,
            std2=atom_width,
            scale=atom_weights[i],
            rotation=0,
        )
        v_map.add_function(func)
    return v_map


# pylint: disable=too-many-locals
def _add_gaussians_for_bonds(
    mol: Chem.Mol,
    conf: Chem.Conformer,
    v_map: ValueGrid,
    bond_weights: npt.NDArray[np.float64],
    bond_width: float,
    bond_length: float,
) -> ValueGrid:
    """Add Gauss-functions centered at bonds to the grid.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule object.
    conf: Chem.Conformer
        Conformation of the molecule.
    v_map: ValueGrid
        ValueGrid object to which the functions are added.
    bond_weights: npt.NDArray[np.float64]
        Array of weights for bonds.
    bond_width: float
        Width of the displayed bond weights (perpendicular to bond-axis).
    bond_length: float
        Length of the displayed bond weights (along the bond-axis).

    Returns
    -------
    ValueGrid
        ValueGrid object with added functions.
    """
    # Adding Gauss-functions centered at bonds (position between the two bonded-atoms)
    for i, b in enumerate(mol.GetBonds()):
        if bond_weights[i] == 0:
            continue
        a1 = b.GetBeginAtom().GetIdx()
        a1_pos = conf.GetAtomPosition(a1)
        a1_coords = np.array([a1_pos.x, a1_pos.y])

        a2 = b.GetEndAtom().GetIdx()
        a2_pos = conf.GetAtomPosition(a2)
        a2_coords = np.array([a2_pos.x, a2_pos.y])

        diff = a2_coords - a1_coords
        angle = np.arctan2(diff[0], diff[1])

        bond_center = (a1_coords + a2_coords) / 2

        func = GaussFunctor2D(
            center=bond_center,
            std1=bond_width,
            std2=bond_length,
            scale=bond_weights[i],
            rotation=angle,
        )
        v_map.add_function(func)
    return v_map


def mapvalues2mol(
    mol: Chem.Mol,
    atom_weights: Sequence[float] | npt.NDArray[np.float64] | None = None,
    bond_weights: Sequence[float] | npt.NDArray[np.float64] | None = None,
    atom_width: float = 0.3,
    bond_width: float = 0.25,
    bond_length: float = 0.5,
    canvas: rdMolDraw2D.MolDraw2D | None = None,
    grid_resolution: Sequence[int] | None = None,
    value_lims: Sequence[float] | None = None,
    color: str | Colormap = "bwr",
    padding: Sequence[float] | None = None,
) -> rdMolDraw2D:
    """Map weights of atoms and bonds to the drawing of a RDKit molecular depiction.

    For each atom and bond of depicted molecule a Gauss-function, centered at the respective object, is created and
    scaled by the corresponding weight. Gauss-functions of atoms are circular, while Gauss-functions of bonds can be
    distorted along the bond axis. The value of each pixel is determined as the sum of all function-values at the pixel
    position. Subsequently, the values are mapped to a color and drawn onto the canvas.

    Inspired from https://github.com/c-feldmann/rdkit_heatmaps/blob/master/rdkit_heatmaps/molmapping.py

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule object which is displayed.
    atom_weights: Sequence[float] | npt.NDArray[np.float64] | None
        Array of weights for atoms.
    bond_weights: Sequence[float] | npt.NDArray[np.float64] | None
        Array of weights for bonds.
    atom_width: float
        Value for the width of displayed atom weights.
    bond_width: float
        Value for the width of displayed bond weights (perpendicular to bond-axis).
    bond_length: float
        Value for the length of displayed bond weights (along the bond-axis).
    canvas: rdMolDraw2D.MolDraw2D | None
        RDKit canvas the molecule and heatmap are drawn onto.
    grid_resolution: Sequence[int] | None
        Number of pixels of x- and y-axis.
    value_lims: Sequence[float] | None
        Lower and upper limit of displayed values. Values exceeding limit are displayed as maximum (or minimum) value.
    color: str | Colormap
        Matplotlib colormap or string referring to a matplotlib colormap
    padding: Sequence[float] | None
        Increase of heatmap size, relative to size of molecule. Usually the heatmap is increased by 100% in each axis
        by padding 50% in each side.

    Returns
    -------
    rdMolDraw2D.MolDraw2D
        Drawing of molecule and corresponding heatmap.
    """
    # assign default values and convert to numpy array
    if atom_weights is None:
        atom_weights = np.zeros(len(mol.GetAtoms()))
    elif not isinstance(atom_weights, np.ndarray):
        atom_weights = np.array(atom_weights)

    if bond_weights is None:
        bond_weights = np.zeros(len(mol.GetBonds()))
    elif not isinstance(bond_weights, np.ndarray):
        bond_weights = np.array(bond_weights)

    if not canvas:
        canvas = rdMolDraw2D.MolDraw2DCairo(800, 450)
        draw_opt = canvas.drawOptions()
        draw_opt.padding = 0.2
        draw_opt.bondLineWidth = 3
        canvas.SetDrawOptions(draw_opt)

    if grid_resolution is None:
        grid_resolution = [canvas.Width(), canvas.Height()]

    if padding is None:
        # take padding from DrawOptions
        draw_opt = canvas.drawOptions()
        padding = [draw_opt.padding * 2, draw_opt.padding * 2]

    # validate input
    if not len(atom_weights) == len(mol.GetAtoms()):
        raise ValueError("len(atom_weights) is not equal to number of bonds in mol")

    if not len(bond_weights) == len(mol.GetBonds()):
        raise ValueError("len(bond_weights) is not equal to number of bonds in mol")

    # extract the 2D conformation of the molecule to be drawn
    conf = mol.GetConformer(0)

    # setup grid and add functions for atoms and bonds
    v_map = _make_grid(mol, canvas, grid_resolution, padding)
    v_map = _add_gaussians_for_atoms(mol, conf, v_map, atom_weights, atom_width)
    v_map = _add_gaussians_for_bonds(
        mol, conf, v_map, bond_weights, bond_width, bond_length
    )

    # evaluate all functions at pixel positions to obtain pixel values
    v_map.evaluate()

    # create color-grid from the value grid.
    c_grid = v_map.map2color(color, v_lim=value_lims)
    # draw the molecule and erase it to initialize the grid
    canvas.DrawMolecule(mol)
    canvas.ClearDrawing()
    # add the Colormap to the canvas
    color_canvas(canvas, c_grid)
    # add the molecule to the canvas
    canvas.DrawMolecule(mol)
    return canvas


def structure_heatmap(
    mol: RDKitMol,
    atom_weights: npt.NDArray[np.float64],
    color_tuple: tuple[RGBAtuple, RGBAtuple, RGBAtuple] | None = None,
    width: int = 600,
    height: int = 600,
) -> Draw.MolDraw2D:
    """Create a Gaussian plot on the molecular structure, highlight atoms with weighted Gaussians.

    Parameters
    ----------
    mol: RDKitMol
        The molecule.
    atom_weights: npt.NDArray[np.float64]
        The atom weights.
    color_tuple: Tuple[RGBAtuple, RGBAtuple, RGBAtuple]
        The color tuple.
    width: int
        The width of the image in number of pixels.
    height: int
        The height of the image in number of pixels.

    Returns
    -------
    Draw.MolDraw2D
        The configured drawer.
    """
    drawer = Draw.MolDraw2DCairo(width, height)
    # Coloring atoms of element 0 to 100 black
    drawer.drawOptions().updateAtomPalette({i: (0, 0, 0, 1) for i in range(100)})
    draw_opt = drawer.drawOptions()
    draw_opt.padding = 0.2

    if color_tuple is None:
        coolwarm = (
            (0.017, 0.50, 0.850, 0.5),
            (1.0, 1.0, 1.0, 0.5),
            (1.0, 0.25, 0.0, 0.5),
        )
        color_tuple = coolwarm

    color_map = color_tuple_to_colormap(color_tuple)

    mol_copy = Chem.Mol(mol)
    mol_copy = Draw.PrepareMolForDrawing(mol_copy)
    mapvalues2mol(
        mol_copy,
        atom_weights=atom_weights,
        bond_weights=None,
        atom_width=0.5,  # 0.4
        bond_width=0.25,
        bond_length=0.5,
        canvas=drawer,
        grid_resolution=None,
        value_lims=None,
        color=color_map,
        padding=None,
    )

    drawer.FinishDrawing()
    return drawer


def show_png(data: bytes) -> Image.Image:
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
