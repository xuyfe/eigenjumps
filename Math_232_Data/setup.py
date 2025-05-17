from setuptools import setup, find_packages

## TO RUN THIS IN DEVELOPMENT MODE: pip install -e .

setup(
    name="math_232_data",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "scikit-learn",
    ],
)

