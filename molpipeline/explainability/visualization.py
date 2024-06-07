"""Visualization functions for the explainability module."""

from __future__ import annotations

import io

import numpy as np
import numpy.typing as npt
from PIL import Image
from rdkit import Geometry
from rdkit.Chem import Draw

from molpipeline.abstract_pipeline_elements.core import RDKitMol

RNGATuple = tuple[float, float, float, float]


def get_similaritymap_from_weights(
    mol: RDKitMol,
    weights: npt.NDArray[np.float_] | list[float] | tuple[float],
    draw2d: Draw.MolDraw2DCairo,
    sigma: float | None = None,
    sigma_f: float = 0.3,
    contour_lines: int = 10,
    contour_params: Draw.ContourParams | None = None,
) -> Draw.MolDraw2D:
    """Generate the similarity map for a molecule given the atomic weights.

    Strongly inspired from Chem.Draw.SimilarityMaps.

    Parameters
    ----------
    mol: RDKitMol
        The molecule of interest.
    weights: Union[npt.NDArray[np.float_], List[float], Tuple[float]]
        The atomic weights.
    draw2d: Draw.MolDraw2DCairo
        The drawer.
    sigma: Optional[float]
        The sigma value.
    sigma_f: float
        The sigma factor.
    contour_lines: int
        The number of contour lines.
    contour_params: Optional[Draw.ContourParams]
        The contour parameters.

    Returns
    -------
    Draw.MolDraw2D
        The drawer.
    """
    if mol.GetNumAtoms() < 2:
        raise ValueError("too few atoms")
    mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
    if not mol.GetNumConformers():
        Draw.rdDepictor.Compute2DCoords(mol)
    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = (
                sigma_f
                * (
                    mol.GetConformer().GetAtomPosition(idx1)
                    - mol.GetConformer().GetAtomPosition(idx2)
                ).Length()
            )
        else:
            sigma = (
                sigma_f
                * (
                    mol.GetConformer().GetAtomPosition(0)
                    - mol.GetConformer().GetAtomPosition(1)
                ).Length()
            )
        sigma = round(sigma, 2)
    sigmas = [sigma] * mol.GetNumAtoms()
    locs = []
    for i in range(mol.GetNumAtoms()):
        atom_pos = mol.GetConformer().GetAtomPosition(i)
        locs.append(Geometry.Point2D(atom_pos.x, atom_pos.y))
    draw2d.DrawMolecule(mol)
    draw2d.ClearDrawing()
    if not contour_params:
        contour_params = Draw.ContourParams()
        contour_params.fillGrid = True
        contour_params.gridResolution = 0.1
        contour_params.extraGridPadding = 0.5
    Draw.ContourAndDrawGaussians(
        draw2d, locs, weights, sigmas, nContours=contour_lines, params=contour_params
    )
    draw2d.drawOptions().clearBackground = False
    draw2d.DrawMolecule(mol)
    return draw2d


def rdkit_gaussplot(
    mol: RDKitMol,
    weights: npt.NDArray[np.float_],
    n_contour_lines: int = 5,
    color_tuple: tuple[RNGATuple, RNGATuple, RNGATuple] | None = None,
) -> Draw.MolDraw2D:
    """Create a Gaussian plot on the molecular structure, highlight atoms with weighted Gaussians.

    Parameters
    ----------
    mol: RDKitMol
        The molecule.
    weights: npt.NDArray[np.float_]
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

    cps.setColourMap(color_tuple)

    drawer = get_similaritymap_from_weights(
        mol,
        weights,
        contour_lines=n_contour_lines,
        draw2d=drawer,
        contour_params=cps,
        sigma_f=0.4,
    )
    # from rdkit.Chem.Draw import SimilarityMaps
    # drawer = SimilarityMaps.GetSimilarityMapFromWeights(
    #     mol,
    #     weights,
    #     contour_lines=n_contour_lines,
    #     draw2d=drawer,
    #     contour_params=cps,
    #     sigma_f=0.4,
    # )
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
