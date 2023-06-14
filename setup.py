"""Everything relevant to install molpipeline."""
from setuptools import setup, find_packages

requirement_list = open("requirements.txt").read().splitlines()

setup(
    name="molpipeline",
    version="0.3.5",
    packages=find_packages(),
    url="",
    license="",
    author="Christian W. Feldmann",
    author_email="christian-wolfgang.feldmann@basf.com",
    description="",
    install_requires=requirement_list,
)
