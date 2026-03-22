from setuptools import setup, find_packages

setup(
    name="maniskill-server",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "torch", "mani_skill"],
)
