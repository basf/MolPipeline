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
from matplotlib.colors import Colormap
from rdkit import Geometry, Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.colors import ListedColormap

from molpipeline.abstract_pipeline_elements.core import RDKitMol
from molpipeline.explainability.visualization.gauss import GaussFunction2D
from molpipeline.explainability.visualization.heatmaps import color_canvas, ValueGrid

RNGATuple = tuple[float, float, float, float]
#
#
# def get_similaritymap_from_weights(
#     mol: RDKitMol,
#     weights: npt.NDArray[np.float64] | list[float] | tuple[float],
#     draw2d: Draw.MolDraw2DCairo,
#     sigma: float | None = None,
#     sigma_f: float = 0.3,
#     contour_lines: int = 10,
#     contour_params: Draw.ContourParams | None = None,
# ) -> Draw.MolDraw2D:
#     """Generate the similarity map for a molecule given the atomic weights.
#
#     Strongly inspired from Chem.Draw.SimilarityMaps.
#
#     Parameters
#     ----------
#     mol: RDKitMol
#         The molecule of interest.
#     weights: Union[npt.NDArray[np.float64], List[float], Tuple[float]]
#         The atomic weights.
#     draw2d: Draw.MolDraw2DCairo
#         The drawer.
#     sigma: Optional[float]
#         The sigma value.
#     sigma_f: float
#         The sigma factor.
#     contour_lines: int
#         The number of contour lines.
#     contour_params: Optional[Draw.ContourParams]
#         The contour parameters.
#
#     Returns
#     -------
#     Draw.MolDraw2D
#         The drawer.
#     """
#     if mol.GetNumAtoms() < 2:
#         raise ValueError("too few atoms")
#     mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
#     if not mol.GetNumConformers():
#         Draw.rdDepictor.Compute2DCoords(mol)
#     if sigma is None:
#         if mol.GetNumBonds() > 0:
#             bond = mol.GetBondWithIdx(0)
#             idx1 = bond.GetBeginAtomIdx()
#             idx2 = bond.GetEndAtomIdx()
#             sigma = (
#                 sigma_f
#                 * (
#                     mol.GetConformer().GetAtomPosition(idx1)
#                     - mol.GetConformer().GetAtomPosition(idx2)
#                 ).Length()
#             )
#         else:
#             sigma = (
#                 sigma_f
#                 * (
#                     mol.GetConformer().GetAtomPosition(0)
#                     - mol.GetConformer().GetAtomPosition(1)
#                 ).Length()
#             )
#         sigma = round(sigma, 2)
#     sigmas = [sigma] * mol.GetNumAtoms()
#     locs = []
#     for i in range(mol.GetNumAtoms()):
#         atom_pos = mol.GetConformer().GetAtomPosition(i)
#         locs.append(Geometry.Point2D(atom_pos.x, atom_pos.y))
#     draw2d.DrawMolecule(mol)
#     draw2d.ClearDrawing()
#     if not contour_params:
#         contour_params = Draw.ContourParams()
#         contour_params.fillGrid = True
#         contour_params.gridResolution = 0.1
#         contour_params.extraGridPadding = 0.5
#     Draw.ContourAndDrawGaussians(
#         draw2d, locs, weights, sigmas, nContours=contour_lines, params=contour_params
#     )
#     draw2d.drawOptions().clearBackground = False
#     draw2d.DrawMolecule(mol)
#     return draw2d


def get_mol_lims(mol: Chem.Mol) -> tuple[tuple[float, float], tuple[float, float]]:
    """Returns the extent of the molecule.

    x- and y-coordinates of all atoms in the molecule are accessed, returning min- and max-values for both axes.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit Molecule object of which the limits are determined.

    Returns
    -------
    Tuple[Tuple[float, float], Tuple[float, float]]
        Limits of the molecule.
    """
    coords = []
    conf = mol.GetConformer(0)
    for i, _ in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append((pos.x, pos.y))
    coords = np.array(coords)
    min_p = np.min(coords, axis=0)
    max_p = np.max(coords, axis=0)
    x_lim = min_p[0], max_p[0]
    y_lim = min_p[1], max_p[1]
    return x_lim, y_lim


def pad(lim: Sequence[float] | npt.NDArray, ratio: float) -> tuple[float, float]:
    """Takes a 2 dimensional vector and adds len(vector) * ratio / 2 to each side and returns obtained vector.

    Parameters
    ----------
    lim: Sequence[float]

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


def color_tuple_to_colormap(color_tuple) -> Colormap:

    # coolwarm = ((0.017, 0.50, 0.850, 0.5), (1.0, 1.0, 1.0, 0.5), (1.0, 0.25, 0.0, 0.5))

    if len(color_tuple) != 3:
        raise ValueError("Color tuple must have 3 elements")

    # Definition of color
    col1, col2, col3 = map(np.array, color_tuple)
    # yellow = np.array([1, 1, 0, 1])
    # white = np.array([1, 1, 1, 1])
    # purple = np.array([1, 0, 1, 1])

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


def mapvalues2mol(
    mol: Chem.Mol,
    atom_weights: Sequence[float] | npt.NDArray | None = None,
    bond_weights: Sequence[float] | npt.NDArray | None = None,
    atom_width: float = 0.3,
    bond_width: float = 0.25,
    bond_length: float = 0.5,
    canvas: rdMolDraw2D.MolDraw2D | None = None,
    grid_resolution=None,
    value_lims: Sequence[float] | None = None,
    color: str | Colormap = "bwr",
    padding: Sequence[float] | None = None,
) -> rdMolDraw2D:
    """A function to map weights of atoms and bonds to the drawing of a RDKit molecular depiction.

    For each atom and bond of depicted molecule a Gauss-function, centered at the respective object, is created and
    scaled by the corresponding weight. Gauss-functions of atoms are circular, while Gauss-functions of bonds can be
    distorted along the bond axis. The value of each pixel is determined as the sum of all function-values at the pixel
    position. Subsequently, the values are mapped to a color and drawn onto the canvas.

    Inspired from https://github.com/c-feldmann/rdkit_heatmaps/blob/master/rdkit_heatmaps/molmapping.py

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule object which is displayed.
    atom_weights: Optional[Union[Sequence[float], np.ndarray]]
        Array of weights for atoms.
    bond_weights: Optional[Union[Sequence[float], np.ndarray]]
        Array of weights for bonds.
    atom_width: float
        Value for the width of displayed atom weights.
    bond_width: float
        Value for the width of displayed bond weights (perpendicular to bond-axis).
    bond_length: float
        Value for the length of displayed bond weights (along the bond-axis).
    canvas: Optional[rdMolDraw2D.MolDraw2D]
        RDKit canvas the molecule and heatmap are drawn onto.
    grid_resolution: Optional[Sequence[int]]
        Number of pixels of x- and y-axis.
    value_lims: Optional[Sequence[float]]
        Lower and upper limit of displayed values. Values exceeding limit are displayed as maximum (or minimum) value.
    color: Union[str, Colormap]
        Matplotlib colormap or string referring to a matplotlib colormap
    padding: Optional[Sequence[float]]
        Increase of heatmap size, relative to size of molecule. Usually the heatmap is increased by 100% in each axis
        by padding 50% in each side.

    Returns
    -------
    rdMolDraw2D.MolDraw2D
        Drawing of molecule and corresponding heatmap.
    """

    # Assigning default values
    if atom_weights is None:
        atom_weights = np.zeros(len(mol.GetAtoms()))

    if bond_weights is None:
        bond_weights = np.zeros(len(mol.GetBonds()))

    if not canvas:
        canvas = rdMolDraw2D.MolDraw2DCairo(800, 450)
        draw_opt = canvas.drawOptions()
        draw_opt.padding = 0.2
        draw_opt.bondLineWidth = 3
        canvas.SetDrawOptions(draw_opt)

    if grid_resolution is None:
        grid_resolution = [canvas.Width(), canvas.Height()]

    if padding is None:
        # Taking padding from DrawOptions
        draw_opt = canvas.drawOptions()
        padding = [draw_opt.padding * 2, draw_opt.padding * 2]

    # Validating input
    if not len(atom_weights) == len(mol.GetAtoms()):
        raise ValueError("len(atom_weights) is not equal to number of bonds in mol")

    if not len(bond_weights) == len(mol.GetBonds()):
        raise ValueError("len(bond_weights) is not equal to number of bonds in mol")

    # Setting up the grid
    xl, yl = get_mol_lims(mol)  # Limit of molecule
    xl, yl = list(xl), list(yl)

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

    xl = pad(xl, padding[0])  # Increasing size of x-axis
    yl = pad(yl, padding[1])  # Increasing size of y-axis
    v_map = ValueGrid(xl, yl, grid_resolution[0], grid_resolution[1])

    conf = mol.GetConformer(0)

    # Adding Gauss-functions centered at atoms
    for i, _ in enumerate(mol.GetAtoms()):
        if atom_weights[i] == 0:
            continue
        pos = conf.GetAtomPosition(i)
        coords = pos.x, pos.y
        f = GaussFunction2D(
            center=coords,
            std1=atom_width,
            std2=atom_width,
            scale=atom_weights[i],
            rotation=0,
        )
        v_map.add_function(f)

    # Adding Gauss-functions centered at bonds (position between the two bonded-atoms)
    for i, b in enumerate(mol.GetBonds()):  # type: Chem.Bond
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

        f = GaussFunction2D(
            center=bond_center,
            std1=bond_width,
            std2=bond_length,
            scale=bond_weights[i],
            rotation=angle,
        )
        v_map.add_function(f)

    # Evaluating all functions at pixel positions to obtain pixel values
    v_map.evaluate()

    # Greating color-grid from the value grid.
    c_grid = v_map.map2color(color, v_lim=value_lims)
    # Drawing the molecule and erasing it to initialize the grid
    canvas.DrawMolecule(mol)
    canvas.ClearDrawing()
    # Adding the Colormap to the canvas
    color_canvas(canvas, c_grid)
    # Adding the molecule to the canvas
    canvas.DrawMolecule(mol)
    return canvas


def structure_heatmap(
    mol: RDKitMol,
    weights: npt.NDArray[np.float64],
    n_contour_lines: int = 5,
    color_tuple: tuple[RNGATuple, RNGATuple, RNGATuple] | None = None,
) -> Draw.MolDraw2D:
    """Create a Gaussian plot on the molecular structure, highlight atoms with weighted Gaussians.

    Parameters
    ----------
    mol: RDKitMol
        The molecule.
    weights: npt.NDArray[np.float64]
        The weights.
    n_contour_lines: int
        The number of contour lines.
    color_tuple: Tuple[RNGATuple, RNGATuple, RNGATuple]
        The color tuple.

    Returns
    -------
    Draw.MolDraw2D
        The configured drawer.
    """
    drawer = Draw.MolDraw2DCairo(600, 600)
    # Coloring atoms of element 0 to 100 black
    drawer.drawOptions().updateAtomPalette({i: (0, 0, 0, 1) for i in range(100)})
    cps = Draw.ContourParams()
    cps.fillGrid = True
    cps.gridResolution = 0.02
    cps.extraGridPadding = 1.2
    coolwarm = ((0.017, 0.50, 0.850, 0.5), (1.0, 1.0, 1.0, 0.5), (1.0, 0.25, 0.0, 0.5))

    if color_tuple is None:
        color_tuple = coolwarm

    color_map = color_tuple_to_colormap(color_tuple)

    # cps.setColourMap(color_tuple)
    # drawer = get_similaritymap_from_weights(
    #     mol,
    #     weights,
    #     contour_lines=n_contour_lines,
    #     draw2d=drawer,
    #     contour_params=cps,
    #     sigma_f=0.4,
    # )

    mol_copy = Chem.Mol(mol)
    mol_copy = Draw.PrepareMolForDrawing(mol_copy)
    mapvalues2mol(
        mol_copy,
        atom_weights=weights,
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
