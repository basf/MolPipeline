"""Everything relevant to install molpipeline."""
from setuptools import setup, find_packages

with open("requirements.txt", mode="r", encoding="utf-8") as requirement_file:
    requirement_list = requirement_file.read().splitlines()

setup(
    name="molpipeline",
    version="0.4.3",
    packages=find_packages(),
    url="",
    license="",
    author="Christian W. Feldmann",
    author_email="christian-wolfgang.feldmann@basf.com",
    description="",
    install_requires=requirement_list,
)
