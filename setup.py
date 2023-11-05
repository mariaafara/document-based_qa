from setuptools import setup, find_packages

# Read the requirements from requirements.txt file
with open("requirements.txt") as f:
    install_requires = f.read().strip().split('\n')

setup(
    name="document-based_qa",
    version="0.1",
    packages=find_packages(),
    install_requires=install_requires,  # Use the dependencies from requirements.txt
)
