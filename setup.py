from setuptools import setup, find_packages
from typing import List

HYPNENATE = '-e .'
def get_requirements(file_path:str)-> List[str]:
    '''This function will return the list of requirements '''
    with open(file_path) as f:
        requirements = f.readlines()
    requirements = [req.replace("\n","") for req in requirements]
    if HYPNENATE in requirements:
        requirements.remove(HYPNENATE)
    return requirements
setup(
    name='mlproject',
    version='0.0.1',
    author='Darshan',
    author_email='dluhadia123@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)