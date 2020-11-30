import setuptools

with open("requirements.txt", mode="r") as requirements_file:
    requirements = [package.split("#")[0] for package in requirements_file.read().split("\n") if not package.startswith("#")]

setuptools.setup(
    name="matrix_multiplication",
    version="0.0.1",
    packages=setuptools.find_packages(),
    install_requires=requirements
)
