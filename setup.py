from setuptools import setup, find_packages
from codecs     import open
from os         import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="sparselab",
    version = "0.0.1",
    description = "a Library for Interferometric Imaging using Sparse Modeling",
    long_description = long_description,

    url = "https://eht-jp.github.io/sparselab",
    author = "Kazu Akiyama",
    author_email = "kakiyama@mit.edu",
    license = "MIT",

    keywords = "imaging astronomy EHT",

    packages = find_packages(exclude=["doc*", "test*"]),
    install_requires = ["astropy", "matplotlib", "numpy", "pandas", "scikit-image", "scipy", "xarray"]
)
