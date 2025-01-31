"""Visualization functions for the explainability module.

Much of the visualization code in this file originates from projects of Christian W. Feldmann:
    https://github.com/c-feldmann/rdkit_heatmaps
    https://github.com/c-feldmann/compchemkit
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from molpipeline.abstract_pipeline_elements.core import RDKitMol
from molpipeline.experimental.explainability.explanation import (
    SHAPFeatureAndAtomExplanation,
)
from molpipeline.experimental.explainability.visualization.gauss import GaussFunctor2D
from molpipeline.experimental.explainability.visualization.heatmaps import (
    ValueGrid,
    color_canvas,
    get_color_normalizer_from_data,
)
from molpipeline.experimental.explainability.visualization.utils import (
    RGBAtuple,
    calc_present_and_absent_shap_contributions,
    get_atom_coords_of_bond,
    get_color_map_from_input,
    get_mol_lims,
    pad,
    plt_to_pil,
    to_png,
)


def _make_grid_from_mol(
    mol: Chem.Mol,
    grid_resolution: Sequence[int],
    padding: Sequence[float],
) -> ValueGrid:
    """Create a grid for the molecule.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule object.
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

    height_to_width_ratio_mol = mol_height / (1e-16 + mol_width)
    # the grids height / weight is the canvas height / width
    height_to_width_ratio_canvas = grid_resolution[1] / grid_resolution[0]

    if height_to_width_ratio_mol < height_to_width_ratio_canvas:
        mol_height_new = height_to_width_ratio_canvas * mol_width
        yl[0] -= (mol_height_new - mol_height) / 2
        yl[1] += (mol_height_new - mol_height) / 2
    else:
        mol_width_new = grid_resolution[0] / grid_resolution[1] * mol_height
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
    for i in range(mol.GetNumAtoms()):
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


def _add_gaussians_for_bonds(
    mol: Chem.Mol,
    conf: Chem.Conformer,
    v_map: ValueGrid,
    bond_weights: npt.NDArray[np.float64],
    bond_width: float,
    bond_length: float,
) -> ValueGrid:  # pylint: disable=too-many-locals
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
    for i, bond in enumerate(mol.GetBonds()):
        if bond_weights[i] == 0:
            continue

        a1_coords, a2_coords = get_atom_coords_of_bond(bond, conf)

        diff = a2_coords - a1_coords
        angle = np.arctan2(diff[0], diff[1])

        bond_center = (a1_coords + a2_coords) / 2

        func = GaussFunctor2D(
            center=bond_center,  # type: ignore
            std1=bond_width,
            std2=bond_length,
            scale=bond_weights[i],
            rotation=angle,
        )
        v_map.add_function(func)
    return v_map


def make_sum_of_gaussians_grid(
    mol: Chem.Mol,
    grid_resolution: Sequence[int],
    padding: Sequence[float],
    atom_weights: Sequence[float] | npt.NDArray[np.float64] | None = None,
    bond_weights: Sequence[float] | npt.NDArray[np.float64] | None = None,
    atom_width: float = 0.3,
    bond_width: float = 0.25,
    bond_length: float = 0.5,
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
    grid_resolution: Sequence[int]
        Number of pixels of x- and y-axis.
    padding: Sequence[float]
        Increase of heatmap size, relative to size of molecule.
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

    Returns
    -------
    rdMolDraw2D.MolDraw2D
        Drawing of molecule and corresponding heatmap.
    """
    # assign default values and convert to numpy array
    if atom_weights is None:
        atom_weights = np.zeros(mol.GetNumAtoms())
    elif not isinstance(atom_weights, np.ndarray):
        atom_weights = np.array(atom_weights)

    if bond_weights is None:
        bond_weights = np.zeros(len(mol.GetBonds()))
    elif not isinstance(bond_weights, np.ndarray):
        bond_weights = np.array(bond_weights)

    # validate input
    if not len(atom_weights) == mol.GetNumAtoms():
        raise ValueError("len(atom_weights) is not equal to number of atoms in mol")

    if not len(bond_weights) == len(mol.GetBonds()):
        raise ValueError("len(bond_weights) is not equal to number of bonds in mol")

    # extract the 2D conformation of the molecule to be drawn
    conf = mol.GetConformer(0)

    # setup grid and add functions for atoms and bonds
    value_grid = _make_grid_from_mol(mol, grid_resolution, padding)
    value_grid = _add_gaussians_for_atoms(
        mol, conf, value_grid, atom_weights, atom_width
    )
    value_grid = _add_gaussians_for_bonds(
        mol, conf, value_grid, bond_weights, bond_width, bond_length
    )

    # evaluate all functions at pixel positions to obtain pixel values
    value_grid.evaluate()

    return value_grid


def _add_shap_present_absent_features_text(
    fig: Figure,
    explanation: SHAPFeatureAndAtomExplanation,
    sum_present_shap: float,
    sum_absent_shap: float,
) -> None:
    """Add text to the figure to display the SHAP prediction composition.

    The added text includes the prediction value, the expected value, the sum of the SHAP values for present features,
    and the sum of the SHAP values for absent features.

    Parameters
    ----------
    fig: Figure
        The figure.
    explanation: SHAPFeatureAndAtomExplanation
        The SHAP explanation.
    sum_present_shap: float
        The sum of the SHAP values for present features.
    sum_absent_shap: float
        The sum of the SHAP values for absent features.
    """
    if explanation.prediction is None:
        raise AssertionError("Prediction value is None.")
    if explanation.expected_value is None:
        raise AssertionError("Expected value is None.")

    color1 = "black"
    color2 = "green"
    color3 = "darkorchid"

    fontsize_numbers = 11
    delta = 0.04
    offset = 0.375
    fig.text(
        offset + delta,
        0.18,
        f"{explanation.prediction[-1]:.2f} =",
        fontsize=fontsize_numbers,
        ha="center",
    )
    fig.text(
        offset + 2 * delta,
        0.18,
        f" {'' if explanation.expected_value[-1] >= 0 else '-'}",
        ha="center",
        fontsize=fontsize_numbers,
        color=color1,
    )
    fig.text(
        offset + 3 * delta,
        0.18,
        f" {abs(explanation.expected_value[-1]):.2f}",
        ha="center",
        fontsize=fontsize_numbers,
        color=color1,
    )
    fig.text(
        offset + 4 * delta,
        0.18,
        f" {'+' if sum_present_shap >= 0 else '-'}",
        ha="center",
        fontsize=fontsize_numbers,
        color=color2,
    )
    fig.text(
        offset + 5 * delta,
        0.18,
        f" {abs(sum_present_shap):.2f}",
        ha="center",
        fontsize=fontsize_numbers,
        color=color2,
    )
    fig.text(
        offset + 6 * delta,
        0.18,
        f" {'+' if sum_absent_shap >= 0 else '-'}",
        fontsize=fontsize_numbers,
        ha="center",
        color=color3,
    )
    fig.text(
        offset + 7 * delta,
        0.18,
        f" {abs(sum_absent_shap):.2f}",
        ha="center",
        fontsize=fontsize_numbers,
        color=color3,
    )

    delta = 0.05
    offset = offset + 0.0165
    fig.text(offset, 0.13, "prediction =", ha="center", fontsize=10)
    fig.text(
        offset + 2 * delta,
        0.12,
        "expected\nvalue",
        ha="center",
        fontsize=10,
        color=color1,
    )
    fig.text(offset + 3 * delta, 0.13, " + ", ha="center", fontsize=10, color=color2)
    fig.text(
        offset + 4 * delta,
        0.12,
        "features\npresent",
        ha="center",
        fontsize=10,
        color=color2,
    )
    fig.text(offset + 5 * delta, 0.13, " + ", ha="center", fontsize=10, color=color3)
    fig.text(
        offset + 6 * delta,
        0.12,
        "features\nabsent",
        ha="center",
        fontsize=10,
        color=color3,
    )


def _structure_heatmap(
    mol: RDKitMol,
    atom_weights: npt.NDArray[np.float64],
    color: str | Colormap | tuple[RGBAtuple, RGBAtuple, RGBAtuple] | None = None,
    width: int = 600,
    height: int = 600,
    color_limits: tuple[float, float] | None = None,
) -> tuple[Draw.MolDraw2D, ValueGrid, ValueGrid, colors.Normalize, Colormap]:
    """Create a heatmap of the molecular structure, highlighting atoms with weighted Gaussian's.

    Parameters
    ----------
    mol: RDKitMol
        The molecule.
    atom_weights: npt.NDArray[np.float64]
        The atom weights.
    color: str | Colormap | tuple[RGBAtuple, RGBAtuple, RGBAtuple] | None
        The color map.
    width: int
        The width of the image in number of pixels.
    height: int
        The height of the image in number of pixels.
    color_limits: tuple[float, float] | None
        The color limits.

    Returns
    -------
    Draw.MolDraw2D, ValueGrid, ColorGrid, colors.Normalize, Colormap
        The configured drawer, the value grid, the color grid, the normalizer, and the
        color map.
    """
    drawer = Draw.MolDraw2DCairo(width, height)
    # Coloring atoms of element 0 to 100 black
    drawer.drawOptions().updateAtomPalette({i: (0, 0, 0, 1) for i in range(100)})
    draw_opt = drawer.drawOptions()
    draw_opt.padding = 0.2

    color_map = get_color_map_from_input(color)

    # create the sums of gaussians value grid
    mol_copy = Chem.Mol(mol)
    mol_copy = Draw.PrepareMolForDrawing(mol_copy)
    value_grid = make_sum_of_gaussians_grid(
        mol_copy,
        atom_weights=atom_weights,
        bond_weights=None,
        atom_width=0.5,  # 0.4
        bond_width=0.25,
        bond_length=0.5,
        grid_resolution=[drawer.Width(), drawer.Height()],
        padding=[draw_opt.padding * 2, draw_opt.padding * 2],
    )

    # create color-grid from the value grid.
    if color_limits is None:
        normalizer = get_color_normalizer_from_data(value_grid.values)
    else:
        normalizer = colors.Normalize(vmin=color_limits[0], vmax=color_limits[1])
    color_grid = value_grid.map2color(color_map, normalizer=normalizer)

    # draw the molecule and erase it to initialize the grid
    drawer.DrawMolecule(mol)
    drawer.ClearDrawing()
    # add the Colormap to the canvas
    color_canvas(drawer, color_grid)
    # add the molecule to the canvas
    drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    return drawer, value_grid, color_grid, normalizer, color_map


def structure_heatmap(
    mol: RDKitMol,
    atom_weights: npt.NDArray[np.float64],
    color: str | Colormap | tuple[RGBAtuple, RGBAtuple, RGBAtuple] | None = None,
    width: int = 600,
    height: int = 600,
    color_limits: tuple[float, float] | None = None,
) -> Image.Image:
    """Create a Gaussian plot on the molecular structure, highlight atoms with weighted Gaussians.

    Parameters
    ----------
    mol: RDKitMol
        The molecule.
    atom_weights: npt.NDArray[np.float64]
        The atom weights.
    color: str | Colormap | tuple[RGBAtuple, RGBAtuple, RGBAtuple] | None
        The color map.
    width: int
        The width of the image in number of pixels.
    height: int
        The height of the image in number of pixels.
    color_limits: tuple[float, float] | None
        The color limits.

    Returns
    -------
    Image
        The image as PNG.
    """
    drawer, *_ = _structure_heatmap(
        mol, atom_weights, color, width, height, color_limits
    )
    figure_bytes = drawer.GetDrawingText()
    image = to_png(figure_bytes)
    return image


def structure_heatmap_shap(  # pylint: disable=too-many-branches, too-many-locals
    explanation: SHAPFeatureAndAtomExplanation,
    color: str | Colormap | tuple[RGBAtuple, RGBAtuple, RGBAtuple] | None = None,
    width: int = 600,
    height: int = 600,
    color_limits: tuple[float, float] | None = None,
) -> Image.Image:
    """Create a heatmap of the molecular structure and display SHAP prediction composition.

    Parameters
    ----------
    explanation: SHAPExplanation
        The SHAP explanation.
    color: str | Colormap | tuple[RGBAtuple, RGBAtuple, RGBAtuple] | None
        The color map.
    width: int
        The width of the image in number of pixels.
    height: int
        The height of the image in number of pixels.
    color_limits: tuple[float, float] | None
        The color limits.

    Returns
    -------
    Image
        The image as PNG.
    """
    if explanation.feature_weights is None:
        raise ValueError("Explanation does not contain feature weights.")
    if explanation.feature_vector is None:
        raise ValueError("Explanation does not contain feature_vector.")
    if explanation.expected_value is None:
        raise ValueError("Explanation does not contain expected value.")
    if explanation.prediction is None:
        raise ValueError("Explanation does not contain prediction.")
    if explanation.molecule is None:
        raise ValueError("Explanation does not contain molecule.")
    if explanation.atom_weights is None:
        raise ValueError("Explanation does not contain atom weights.")

    if explanation.prediction.ndim > 2:
        raise ValueError(
            "Unsupported shape for prediction. Maximum 2 dimension is supported."
        )

    if explanation.feature_weights.ndim == 1:
        feature_weights = explanation.feature_weights
    elif explanation.feature_weights.ndim == 2:
        feature_weights = explanation.feature_weights[:, 1]
    else:
        raise ValueError("Unsupported shape for feature weights.")

    # calculate the sum of the SHAP values for present and absent features
    sum_present_shap, sum_absent_shap = calc_present_and_absent_shap_contributions(
        explanation.feature_vector, feature_weights
    )

    with plt.ioff():

        drawer, _, _, normalizer, color_map = _structure_heatmap(
            explanation.molecule,
            explanation.atom_weights,
            color=color,
            width=width,
            height=height,
            color_limits=color_limits,
        )
        figure_bytes = drawer.GetDrawingText()
        image_heatmap = to_png(figure_bytes)
        image_array = np.array(image_heatmap)

        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(
            image_array,
            cmap=color_map,
            norm=normalizer,
        )
        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # remove border
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.015, pad=0.0)

        _add_shap_present_absent_features_text(
            fig, explanation, sum_present_shap, sum_absent_shap
        )

        image = plt_to_pil(fig)
        # clear the figure and memory
        plt.close(fig)

        # remove dpi info because it crashes ipython's display function
        if "dpi" in image.info:
            del image.info["dpi"]
        # keep RDKit's image info
        image.info.update(image_heatmap.info)

    return image
