# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SOARegression",
    version="0.0.3",
    author="Tyler Blume",
    url="https://github.com/tblume1992/SOARegression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description = "Sample Optimized Adaptive Regression.",
    author_email = 't-blume@hotmail.com', 
    keywords = ['machine learning', 'linear regression'],
      install_requires=[           
                        'numpy',
                        'pandas',
                        'scikit-learn',
                        'scipy',
                        'matplotlib',
                        ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


