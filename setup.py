import os
from setuptools import setup, find_packages

setup(
    name='GraphRL',
    version='0.1dev',
    packages=[os.path.join("graphrl", package) for package in find_packages(where="graphrl")],
    long_description=open('README.md').read(),
)
