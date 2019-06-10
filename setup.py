import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyxdh",
    version="0.0.1",
    author="ajz34",
    author_email="17110220038@fudan.edu.cn",
    description="xDH Functional Derivatives Enpowered by Python",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/ajz34/Python-xDH",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Currently Private",
        "Operating System :: Linux",
    ],
)