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
    url="https://github.com/ajz34/Py_xDH",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL v3",
        "Operating System :: Linux",
    ],
    # https://stackoverflow.com/questions/1612733/including-non-python-files-with-setup-py
    package_data={'': ['Validation/*']},
    include_package_data=True,
)
