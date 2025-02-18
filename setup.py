from setuptools import setup, find_packages

setup(
    name="brad-enrichment",
    version="0.1.0",
    author="Joshua Pickard",
    author_email="jpic@umich.edu",
    description="A command-line tool for BRAD enrichment analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JPickard1/BRAD-Enrichment",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "openpyxl",
        "click"  # for CLI
    ],
    entry_points={
        "console_scripts": [
            "brad-enrichment=brad_enrichment.brad_enrichment:cli",
            "brad-builddb=brad_enrichment.enrichment_literature_database:cli"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

