from setuptools import setup

# List of requirements
requirements = [
    'setuptools',
    'torch>=2.3.0',
    'triton>=2.3.0',
    'numpy',
    'matplotlib>=3.9.0',
    'pandas>=2.2.2',
    'tabulate'
    
]

def setup_package():
    setup(
        name='dependency-installer',
        version='0.1.0',
        description='Install project dependencies',
        install_requires=requirements,
        zip_safe=False,
    )

if __name__ == '__main__':
    setup_package()

