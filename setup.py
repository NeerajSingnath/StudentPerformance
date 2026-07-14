from typing import List
from setuptools import find_packages, setup





def get_requirements(path: str) -> List[str]:
    """
    function for reading requirements.txt`
    """
    requirements = []
    with open(path) as f:
        requirements = f.readlines()
    requirements = [req.replace("\n", "") for req in requirements]

    if "-e ." in requirements:
        requirements.remove("-e .")

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Neeraj Singnath",
    author_email="Neerajs604086@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
