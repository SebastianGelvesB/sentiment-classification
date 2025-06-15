# setup.py
from setuptools import setup, find_packages

setup(
    name='sentiment_classification',
    version='0.2.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
