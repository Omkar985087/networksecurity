from setuptools import find_packages,setup
from typing import List

def get_rquirements()->List[str]:
    """
    this function will return lists of requirements
    
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #read lines from the file
            lines=file.readlines()
            #process each line
            for line in lines:
                requirement=line.strip()
                #ignore empty lines and -e .
                if requirement and requirement!='-e .':
                    requirement_lst.append(requirement)
    
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="Netqork Security",
    version="0.0.1",
    author="Omkar Kadam",
    author_email="omkarkadam9422@gmail.com",
    packages=find_packages(),
    install_requires=get_rquirements()
)

