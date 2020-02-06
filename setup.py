import setuptools
import manufacturing

with open("readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="manufacturing",
    version=manufacturing.__version__,
    author="Jason R. Jones",
    author_email="slightlynybbled@gmail.com",
    description="Six-sigma-based analysis of manufacturing "
                "data for trends, Cpk/Ppk.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slightlynybbled/manufacturing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
