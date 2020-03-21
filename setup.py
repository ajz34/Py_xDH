import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as rq:
    # https://stackoverflow.com/questions/26900328/install-dependencies-from-setup-py
    install_requires = rq.read().splitlines()

setuptools.setup(
    name="pyxdh",
    version="v0.0.4",
    author="ajz34",
    author_email="17110220038@fudan.edu.cn",
    description="Document and code of python and PySCF approach XYG3 type of density functional realization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajz34/Py_xDH",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
    # https://stackoverflow.com/questions/1612733/including-non-python-files-with-setup-py
)
