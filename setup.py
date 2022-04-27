from setuptools import setup, find_packages

setup(
    name='marimba',
    version='0.1.0',
    description='Marimba is a Python library for interacting with the Marimba API',
    author='Juan Rocamonde',
    packages=find_packages(),
    scripts=['scripts/marimba'],
)
