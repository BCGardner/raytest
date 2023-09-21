from setuptools import setup, find_packages


def read_requirements(file_name):
    with open(file_name, "r") as file:
        return [line.strip() for line in file if line.strip()]

setup(
    name='raytest',
    version='0.1.0',
    author='Brian Gardner',
    author_email="brgardner@hotmail.co.uk",
    description='Ray testing on the cluster',
    long_description=open('README.md').read(),
    url="https://github.com/BCGardner/raytest",
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.11',
    install_requires=read_requirements("requirements.txt")
)
