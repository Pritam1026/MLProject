from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    """
    This will read the requirments.txt and return the list of packages to be installed.
    
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='Mlproject',
    version='0.0.1',
    author='Pritam',
    author_email='singhpritam983@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)