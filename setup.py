# setup.py file  uses .e in the requirements.txt to make US_VISA_APPROVAL_PREDICTION available as a package
# to other projects.

from setuptools import setup, find_packages

setup(
    name="US_VISA_APPROVAL_PREDICTION",
    version="0.0.0",
    author="Dhruv Tewari",
    author_email="dhrutewa@gmail.com",
    packages=find_packages()
) 