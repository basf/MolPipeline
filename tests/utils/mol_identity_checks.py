"""Functions to check molecular identity and similarity."""

import re


def inchi_str_to_regex(inchi: str) -> str:
    """Convert InChI string to regex pattern for matching.

    Parameters
    ----------
    inchi : str
        The InChI string to convert.

    Returns
    -------
    str
        The regex pattern for matching the InChI string.

    """
    escaped_inchi = re.escape(inchi)
    return f"^{escaped_inchi}"
