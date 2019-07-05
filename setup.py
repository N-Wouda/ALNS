import setuptools


VERSION = "1.0.0"


# Long description is simply the README file
with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="alns",
    version=VERSION,
    author="Niels Wouda",
    author_email="nielswouda@gmail.com",
    description="A flexible implementation of the adaptive large neighbourhood"
                " search (ALNS) algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/N-Wouda/ALNS",
    project_urls={
        "Tracker": "https://github.com/N-Wouda/ALNS/issues",
        "Source": "https://github.com/N-Wouda/ALNS"
    },
    packages=setuptools.find_packages(exclude=['examples']),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha"
    ],
    python_requires='~=3.4',
    install_requires=[
        'numpy >= 1.15.2',
        'matplotlib >=  2.2.0',
    ])
