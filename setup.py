import os
import subprocess
from setuptools import setup, find_packages, Command
from setuptools.command.build_py import build_py as _build_py


class BuildSharedLib(Command):

    description = "Build the C++ shared library using build.sh"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        script_path = os.path.join(os.path.dirname(__file__), "build.sh")
        if not os.path.exists(script_path):
            raise FileNotFoundError("build.sh not found")
        subprocess.call(["bash", script_path])


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name = "nn_package",
    version = "0.1.0",
    author = "Akshat Guduru",
    author_email = "akshat.guduru@gmail.com",
    description = "A C++ neural network package with Python bindings",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages = find_packages(),

    package_data = {'neural_network': ["libnn.*"]},
    include_package_data = True,
    cmdclass = {
        'build_shared' : BuildSharedLib,
        'build_py' : _build_py
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
    install_requires = [
        "numpy",
    ]

)