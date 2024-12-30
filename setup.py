from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove '-e .' if present
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='credit card',  # Ensure no trailing space
    version='0.0.1',
    author='vikash',
    author_email='vikashchauhanvv26@gmaiol.com',
    install_requires=get_requirements('requirements.txt'),  # Get the requirements from the file
    packages=find_packages()
)

