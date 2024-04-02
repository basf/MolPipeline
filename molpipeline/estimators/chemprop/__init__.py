"""Initialize Chemprop module."""

import pkgutil

installed_packages = {pkg.name for pkg in pkgutil.iter_modules()}
if "chemprop" in installed_packages:
    from molpipeline.estimators.chemprop.models import Chemprop  # noqa

    __all__ = ["Chemprop"]
else:
    __all__ = []
